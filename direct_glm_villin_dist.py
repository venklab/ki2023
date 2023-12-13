#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import sqlite3
import numpy as np
import random
import time
import math

from skimage import io
from skimage.morphology import disk, binary_erosion
import utimage
from utmask import masklocation, mask2shape, mask2crop
from utmask import mask2rleblob, rleblob2mask
from direct_visible_villin_dist import update_db_schema
from direct_visible_villin_dist import query_db
from direct_visible_villin_dist import check_and_add_direct
from direct_visible_villin_dist import get_all_masks
from direct_visible_villin_dist import get_mask_neigb
from direct_visible_villin_dist import compress_obj, uncompress_obj

GLM_ID_OFFSET = 30000
MPP = 0.5
# cortex thickness around 2mm, get mask in a rectangle 2mm x 2mm
# 2mm is too big, it may cross to other side of kidney, see KO 1943L
# try 700um x 2
HALF_REGIEON = int(700 / MPP)

"""
Glm and villin segmentations are stored in separated sqlite db. Expected
directory structure for slides:

    -- upper_directory
        --glm
            slide and sqlite files
            ...

        --villin
            slide and sqlite files
            ...
"""


def _put_one_mask_in_region(cvsm, newv, mx, my, mask, w, h):
    """ fast enough, usually use less than 0.0001 second """

    mh, mw = mask.shape
    mx1 = mx + mw - 1
    my1 = my + mh - 1
    mcrop_x = 0
    mcrop_x1 = mw
    needcrop = False
    cvsd_x, cvsd_x1, cvsd_y, cvsd_y1 = mx, mx1, my, my1
    if mx <= 0:
        mcrop_x = -1 * mx
        cvsd_x = 0
        needcrop = True

    if mx1 >= (w - 1):
        mcrop_x1 = mw - (mx1 - (w - 1) + 1)
        cvsd_x1 = w
        needcrop = True

    mcrop_y = 0
    mcrop_y1 = mh
    if my <= 0:
        mcrop_y = -1 * my
        cvsd_y = 0
        needcrop = True

    if my1 >= (h - 1):
        mcrop_y1 = mh - (my1 - (h - 1) + 1)
        cvsd_y1 = h - 1
        needcrop = True

    if needcrop:
        np.copyto(cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1], newv,
                  where=mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1])

    else:
        np.copyto(cvsm[my:my1+1, mx:mx1+1], newv, where=mask)

    return cvsm


def put_glm_mask_on_villin_canvas(cvsm, offset_x, offset_y, db, mid, dzscale):
    """
    offset_x, offset_y is calc on villin canvas
    """
    print('put_glm_mask_on_villin_canvas query mask %d' % mid)
    r = query_db(db,
            'select x,y,w,h,mask from Mask where id=?', (mid,), True)
    mx,my,mw,mh,blob = r
    mask = rleblob2mask(blob)
    if dzscale != 1.0:
        scaled_w = int(float(mw) / dzscale)
        scaled_h = int(float(mh) / dzscale)
        mask = utimage.resize_mask(mask, scaled_w, scaled_h)
    # new x, y, cx, cy after scale
    mx = int(float(mx - offset_x) / dzscale)
    my = int(float(my - offset_y) / dzscale)
    mid += GLM_ID_OFFSET
    h, w = cvsm.shape
    _put_one_mask_in_region(cvsm, mid, mx, my, mask, w, h)


def main():
    # expect parent directory has villin/ and glm/ subdir
    dbf = sys.argv[1]
    dirname = os.path.dirname(dbf)
    bname = os.path.basename(dbf)
    vil_dbf = os.path.abspath(os.path.join(dirname, '../villin', bname))
    glm_dbf = os.path.abspath(os.path.join(dirname, '../glm', bname))

    con_glm = sqlite3.connect(glm_dbf, timeout=5)
    con_vil = sqlite3.connect(vil_dbf, timeout=5)
    update_db_schema(con_glm)
    t0 = time.time()
    dzscale = 2.0

    # crop the whole slide
    sql = 'select min(x), max(x+w), min(y), max(y+h) from Mask'
    r1 = query_db(con_glm, sql, one=True)
    r2 = query_db(con_vil, sql, one=True)
    x0 = min(r1[0], r2[0])
    x1 = max(r1[1], r2[1])
    y0 = min(r1[2], r2[2])
    y1 = max(r1[3], r2[3])
    offset_x, offset_y = x0, y0
    w = x1 - x0 + 1
    h = y1 - y0 + 1

    res = get_all_masks(con_vil, dzscale, offset_x, offset_y, w, h)
    res_glm = get_all_masks(con_glm, dzscale, offset_x, offset_y, w, h)

    t1 = time.time()
    # key is sorted (mid1, mid2) tuple, value is distances found by measure p3
    # p4, or found by dilate mask1 until reaches mask2

    direct_dists = {}
    pairs = {mid+GLM_ID_OFFSET: set([]) for mid in res_glm['mids']}
    for mid in res['mids']:
        pairs[mid] = set([])

    direct_pairs = {'pairs': pairs, 'dists': direct_dists, 'pair_cords': {}}
    print('get_all_masks uses %.2f s' % (t1 - t0))

    offset_x, offset_y = res['offset_x'], res['offset_y']
    for mid in res_glm['mids']:
        # add an offset to make the glm id unlike collide with villin id
        # put glm on to villin canvas
        cvs = res['cvs'].copy()
        put_glm_mask_on_villin_canvas(cvs, offset_x, offset_y, con_glm,
                                      mid, dzscale)

        # convert center to cord on villin canvas
        center = res_glm['orig_centers'][mid]
        cx, cy = center  # orig center
        cx = int(float(cx - offset_x) / dzscale)
        cy = int(float(cy - offset_y) / dzscale)
        pt1 = (cx, cy)  # scale to villin canvas
        mid += GLM_ID_OFFSET

        # get_mask_neigb find neighbors close to center (cord on orig WSI)
        nbs = get_mask_neigb(con_vil, mid, center, half_regieon=HALF_REGIEON)
        for nb_mid in nbs:
            if (nb_mid in pairs[mid]) or (mid in pairs[nb_mid]):
                continue

            pt2 = res['scaled_centers'][nb_mid]
            check_and_add_direct(cvs, mid, nb_mid, pt1, pt2,
                                 direct_pairs, None, dzscale)

        print('%d is direct sight to ' % mid, pairs[mid])

    for mid, nbs in pairs.items():
        tmp = [x for x in nbs]
        pairs[mid] = tmp

    # ztxt1 for villin-villin, ztxt2 for glm-villin
    cur = con_glm.cursor()
    cur.execute('update Info set ztxt2=? where title=?',
            (compress_obj(direct_pairs), 'slideinfo'))

    con_glm.commit()
    print('db commited')


def verify():
    """ plot a random mask and its immediate neighbors """

    dbf = sys.argv[1]
    dirname = os.path.dirname(dbf)
    bname = os.path.basename(dbf)
    vil_dbf = os.path.abspath(os.path.join(dirname, '../villin', bname))

    con_glm = sqlite3.connect(dbf, timeout=5)
    con_vil = sqlite3.connect(vil_dbf, timeout=5)

    row = query_db(con_glm, 'select ztxt2 from Info limit 1', one=True)
    #jbody = json.loads(row[0])['direct_pairs']
    jbody = uncompress_obj(row[0])
    direct_pairs = jbody['pairs']
    dists = jbody['dists']

    rows = query_db(con_glm, 'select id from Mask where is_bad=0')
    all_mids = [r[0] for r in rows]

    i = 0
    dzscale = 2.0

    # crop the whole slide
    sql = 'select min(x), max(x+w), min(y), max(y+h) from Mask'
    r1 = query_db(con_glm, sql, one=True)
    r2 = query_db(con_vil, sql, one=True)
    x0 = min(r1[0], r2[0])
    x1 = max(r1[1], r2[1])
    y0 = min(r1[2], r2[2])
    y1 = max(r1[3], r2[3])
    offset_x, offset_y = x0, y0
    w = x1 - x0 + 1
    h = y1 - y0 + 1

    res_vil = get_all_masks(con_vil, dzscale, offset_x, offset_y, w, h)

    h, w = res_vil['cvs'].shape
    offset_x, offset_y = res_vil['offset_x'], res_vil['offset_y']
    #tgt_mids = [9121, 10062]  # WT Sham
    #for mid in tgt_mids:
    while i < 10:
        cvs = res_vil['cvs']  # .copy()
        print('i = %d' % i)
        mid = random.choice(all_mids)

        cvs3 = np.zeros((h, w, 3), dtype=np.uint8)
        cvsm = np.zeros((h, w), dtype=np.uint8)
        # the resulting cvs has glm mask value of mid + GLM_ID_OFFSET
        put_glm_mask_on_villin_canvas(cvsm, offset_x, offset_y, con_glm,
                                      mid, dzscale)
        cvs_glm_id = mid + GLM_ID_OFFSET
        nbs = direct_pairs[str(cvs_glm_id)]
        print('mask %d has neighbours' % mid, nbs)

        cvs3[:, :, :][cvs != 0] = 100

        cvs3[:, :, 0][cvsm != 0] = 50
        cvs3[:, :, 1][cvsm != 0] = 255
        cvs3[:, :, 2][cvsm != 0] = 50

        mid_dists = []
        for nb_mid in nbs:
            tmp = (cvs_glm_id, nb_mid) if cvs_glm_id < nb_mid else (nb_mid,
                    cvs_glm_id)
            k = '%d_%d' % tmp
            dist = dists[k]
            dist.sort()
            mid_dists.append((dist[-1], nb_mid))

        mid_dists.sort(reverse=True)
        f5 = set([x[1] for x in mid_dists[:5]])
        for nb_mid in nbs:
            if nb_mid in f5:
                color = (255, 255, 0)  # yellow for furthest 5
            else:
                color = (255, 0, 255)  # magenta

            for j in range(3):
                cvs3[:, :, j][cvs == nb_mid] = color[j]

        outfn = '/nfs/wsi/wei/tmp/f5_glm_masks_%d.png' % mid
        io.imsave(outfn, cvs3)
        print('output saved to %s' % outfn)
        i += 1



def verify_border_dist():
    """ plot a mask and border lines to its immediate neighbors on slide """
    import cv2
    dbf = sys.argv[1]
    con = sqlite3.connect(dbf, timeout=5)

    row = query_db(con, 'select ztxt2 from Info limit 1', one=True)
    #jbody = json.loads(row[0])['direct_pairs']
    jbody = uncompress_obj(row[0])
    direct_pairs = jbody['pairs']
    all_mids = [int(x) for x in direct_pairs]
    i = 0
    center_v = 0xffff / 6
    other_v = int(center_v / 1.5)
    #res = get_all_masks(con, dzscale=2.0)
    res = get_all_masks(con, dzscale=1.0)
    h, w = res['cvs'].shape
    # check 1516,1371, 520, 1370, 1431, 1476

    #mid = random.choice(all_mids)
    mid = 5990
    mid = 6321
    cvs = np.copy(res['cvs'])
    cvs2 = np.zeros((h, w, 3), dtype=np.uint8)
    nbs = direct_pairs[str(mid)]
    print('mask %d has neighbours' % mid, nbs)

    cvs2[:, :, :][cvs != 0] = 100

    cvs2[:, :, 0][cvs == mid] = 50
    cvs2[:, :, 1][cvs == mid] = 255
    cvs2[:, :, 2][cvs == mid] = 50

    #mid2 = nbs[3]
    #mid2 = nbs[7]
    # todo plot 5990 6001 in full resolution
    for mid2 in nbs:
        if mid2 != 5813:
            continue

        cvs3 = cvs2.copy()
        cvs3[:, :, 0][cvs == mid2] = 50
        cvs3[:, :, 1][cvs == mid2] = 255
        cvs3[:, :, 2][cvs == mid2] = 50

        tmp = (mid, mid2) if mid < mid2 else (mid2, mid)
        k = '%d_%d' % tmp
        pair_cords = jbody['pair_cords'][k]
        for (pt1, pt2, pt3, pt4) in pair_cords:
            pt1 = scale_point(pt1, 2)
            pt2 = scale_point(pt2, 2)
            pt3 = scale_point(pt3, 2)
            pt4 = scale_point(pt4, 2)

            cv2.line(cvs3, pt1, pt2, (255, 0, 255), 1)
            cv2.circle(cvs3, pt1, 4, (255, 255, 0), -1)
            cv2.circle(cvs3, pt2, 4, (255, 255, 0), -1)
            cv2.line(cvs3, pt3, pt4, (255, 255, 255), 2)

        outfn = '/nfs/wsi/wei/paper/fig1/full_verify_mask_border_line_%d_%d.png' % (
                mid, mid2)
        io.imsave(outfn, cvs3)
        print('output saved to %s' % outfn)
        i += 1


def mask_outline(cvs_in):
    """ expect a binary mask, foreground has non-zero value """
    cvs = cvs_in.copy()
    cvs2 = binary_erosion(cvs, disk(8))
    cvs[cvs2] = 0
    return cvs


def verify_on_slide():
    """ plot a random mask and its immediate neighbors, overlap on slide """
    from utslide import UTSlide
    from PIL import Image

    dbf = sys.argv[1]
    dirname = os.path.dirname(dbf)
    bname = os.path.basename(dbf)
    no_ext = os.path.splitext(bname)[0]
    vil_dbf = os.path.abspath(os.path.join(dirname, '../villin', bname))
    vil_wsi = os.path.abspath(os.path.join(dirname, '../villin',
                              '%s.svs' % no_ext))
    con_glm = sqlite3.connect(dbf, timeout=5)
    con_vil = sqlite3.connect(vil_dbf, timeout=5)

    row = query_db(con_glm, 'select ztxt2 from Info limit 1', one=True)
    #jbody = json.loads(row[0])['direct_pairs']
    jbody = uncompress_obj(row[0])
    direct_pairs = jbody['pairs']
    dists = jbody['dists']

    rows = query_db(con_glm, 'select id from Mask where is_bad=0')
    all_mids = [r[0] for r in rows]

    dzscale = 1.0

    # crop the whole slide
    sql = 'select min(x), max(x+w), min(y), max(y+h) from Mask'
    r1 = query_db(con_glm, sql, one=True)
    r2 = query_db(con_vil, sql, one=True)
    x0 = min(r1[0], r2[0])
    x1 = max(r1[1], r2[1])
    y0 = min(r1[2], r2[2])
    y1 = max(r1[3], r2[3])
    offset_x, offset_y = x0, y0
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    res_vil = get_all_masks(con_vil, dzscale, offset_x, offset_y, w, h)
    h, w = res_vil['cvs'].shape
    offset_x, offset_y = res_vil['offset_x'], res_vil['offset_y']

    slide = UTSlide(vil_wsi)

    # WT_D14_2755L glm
    tgt_mids = [4, 23, 37,47, 90, 106, 125, 130,  134, 176, ]
    tgt_mids = [31, 111, 167, 170]

    # WT_Sham_9296L glm
    tgt_mids = [218, 287, 302, 303, 306, 318, 349, 353, 355, 384]
    tgt_mids =[365,]
    for mid in tgt_mids:
        row = query_db(con_glm, 'select cx,cy from Mask where id=?', (mid,), True)
        cx, cy = row
        # crop the canvas to rectangle centered at (cx, cy)
        cx -= offset_x
        cy -= offset_y
        hwh = 1200  # half width height

        cvs = res_vil['cvs'].copy()
        # the resulting cvs has glm mask value of mid + GLM_ID_OFFSET
        put_glm_mask_on_villin_canvas(cvs, offset_x, offset_y, con_glm,
                                      mid, dzscale)
        x0 = max(0, cx-hwh)
        y0 = max(0, cy-hwh)

        cvs2 = cvs[y0:cy+hwh, x0:cx+hwh]
        cvs_h, cvs_w = cvs2.shape
        # draw mask as alpha
        cvsa = np.full((cvs_h, cvs_w, 4), 0, dtype=np.uint8)

        cvs_glm_id = mid + GLM_ID_OFFSET
        nbs = direct_pairs[str(cvs_glm_id)]
        print('mask %d has neighbours' % mid, nbs)

        cvsm = np.zeros((cvs_h, cvs_w), dtype=bool)

        cvsm[cvs2 == cvs_glm_id] = 1
        outline_cvsm = mask_outline(cvsm)
        color = (50, 255, 50, 255)
        for ch in range(4):
            cvsa[:, :, ch][outline_cvsm] = color[ch]

        mid_dists = []
        for nb_mid in nbs:
            tmp = (cvs_glm_id, nb_mid) if cvs_glm_id < nb_mid else (nb_mid,
                    cvs_glm_id)
            k = '%d_%d' % tmp
            dist = dists[k]
            dist.sort()
            mid_dists.append((dist[-1], nb_mid))

        mid_dists.sort(reverse=True)
        f5 = set([x[1] for x in mid_dists[:5]])
        for nb_mid in nbs:
            if nb_mid in f5:
                color = (255, 255, 0, 255)  # yellow for the furthest
            else:
                color = (0, 0, 255, 255)  # blue for others

            cvsm = np.zeros((cvs_h, cvs_w), dtype=bool)
            cvsm[cvs2 == nb_mid] = 1
            outline_cvsm = mask_outline(cvsm)

            for j in range(4):
                cvsa[:, :, j][outline_cvsm] = color[j]

        other_mids = set(np.unique(cvs2)[1:]) - set(nbs)
        other_mids.remove(cvs_glm_id)
        color = (0, 0, 0, 80)
        for nb_mid in other_mids:
            # gray for others
            cvsm = np.zeros((cvs_h, cvs_w), dtype=bool)
            cvsm[cvs2 == nb_mid] = 1
            outline_cvsm = mask_outline(cvsm)

            for j in range(4):
                cvsa[:, :, j][outline_cvsm] = color[j]

        x = x0 + offset_x
        y = y0 + offset_y
        tmp = slide.read_l0_region((x,y), (cvs_w, cvs_h))

        im = np.asarray(tmp).copy()
        img_rgba = Image.fromarray(im)
        img_mask = Image.fromarray(cvsa)
        img_rgba.alpha_composite(img_mask)
        im = np.array(img_rgba)
        outfn = '/nfs/wsi/wei/tmp/f5_glm_on_slide_%s_%d.png' % (no_ext, mid)
        io.imsave(outfn, im)
        print('output saved to %s' % outfn)


if __name__ == "__main__":
    main()
