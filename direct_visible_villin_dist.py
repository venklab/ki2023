#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import os
import random
import sqlite3
import sys
import time
import zlib

import numpy as np
from skimage import io
from skimage.morphology import disk, binary_erosion

import utimage
from utmask import masklocation, mask2shape, mask2crop
from utmask import mask2rleblob, rleblob2mask


MPP = 0.5
# cortex thickness around 2mm, get mask in a rectangle 2mm x 2mm
# 2mm is too big, it may cross to other side of kidney, see KO 1943L
# try 700um x 2
HALF_REGIEON = int(700 / MPP)

dilate_cache = {}


def compress_obj(obj):
    return zlib.compress(json.dumps(obj).encode('ascii'), level=9)


def uncompress_obj(blob):
    return json.loads(zlib.decompress(blob).decode('ascii'))


def query_db(conn, query, args=(), one=False):
    ''' wrap the db query, fetch into one step '''
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def update_db_schema_step_2(db):
    # update Mask table, add several real value column
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='jtxt1'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column jtxt1 to 4 (date type TEXT) to Mask')
        db.execute('ALTER TABLE Mask ADD COLUMN jtxt1 TEXT;')
        db.execute('ALTER TABLE Mask ADD COLUMN jtxt2 TEXT;')
        db.execute('ALTER TABLE Mask ADD COLUMN jtxt3 TEXT;')
        db.execute('ALTER TABLE Mask ADD COLUMN jtxt4 TEXT;')
        db.commit()

    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='cx'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column cx, cy to 4 (date type INTEGER) to Mask')
        db.executescript('''
ALTER TABLE Mask ADD COLUMN cx INTEGER;
ALTER TABLE Mask ADD COLUMN cy INTEGER;
CREATE INDEX IF NOT EXISTS idx_Mask_cx on Mask (cx);
CREATE INDEX IF NOT EXISTS idx_Mask_cy on Mask (cy);
CREATE INDEX IF NOT EXISTS idx_Mask_x on Mask (x);
CREATE INDEX IF NOT EXISTS idx_Mask_y on Mask (y);''')
        db.commit()

    db.execute('''UPDATE Mask set cx=cast((x+w*0.5+0.5) as integer),
                                  cy=cast((y+h*0.5+0.5) as integer);''')
    db.commit()

    sql = ("SELECT count(*) FROM pragma_table_info('Info') "
           " WHERE name='ztxt1'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column ztxt1 to 4 (date type BLOB) to Info')
        db.execute('ALTER TABLE Info ADD COLUMN ztxt1 BLOB;')
        db.execute('ALTER TABLE Info ADD COLUMN ztxt2 BLOB;')
        db.execute('ALTER TABLE Info ADD COLUMN ztxt3 BLOB;')
        db.execute('ALTER TABLE Info ADD COLUMN ztxt4 BLOB;')
        db.commit()


def update_db_schema(db):
    # update Mask table, add several real value column
    print('check and update db schema...')
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='r1'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        print('update db schema: add column r1 to r4 (date type REAL) to Mask')
        db.execute('ALTER TABLE Mask ADD COLUMN r1 REAL;')
        db.execute('ALTER TABLE Mask ADD COLUMN r2 REAL;')
        db.execute('ALTER TABLE Mask ADD COLUMN r3 REAL;')
        db.execute('ALTER TABLE Mask ADD COLUMN r4 REAL;')
        db.commit()

    update_db_schema_step_2(db)


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
        #cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1][
        #        mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1]] = True
        np.copyto(cvsm[cvsd_y:cvsd_y1+1, cvsd_x:cvsd_x1+1], newv,
                  where=mask[mcrop_y:mcrop_y1+1, mcrop_x:mcrop_x1+1])

    else:
        #cvsm[my:my1+1, mx:mx1+1][mask] = True
        np.copyto(cvsm[my:my1+1, mx:mx1+1], newv, where=mask)

    return cvsm


def find_mask_center(mask, w, h):
    """ center of a mask can be outside. Draw a horizontal line a this center,
    use the center of its intersection with mask as the real center """

    """
    rough estimate mask center by moving a 32x32 from geometric center towards
    left and right, take the first time the rectangle inside mask, or the
    rectangle reaches largest overlap with mask
    """
    cx, cy = int(w / 2.0), int(h / 2.0)
    # draw a horizontal line pass center, find continue line segments
    line_segments = []
    x = 0
    x0, x1 = -1, -1
    prev = 0
    while x < w:
        v = mask[cy, x]
        if x0 == -1 and prev == 0 and v != 0:
            #print('x0 = ', x)
            x0 = x
        elif x1 == -1 and prev != 0 and v == 0:
            #print('x1 = ', x)
            x1 = x

        if x0 >= 0 and x1 >= 0:
            line_segments.append((x0, x1))
            x0, x1 = -1, -1

        prev = v
        x += 1

    if x0 > 0 and x0 != (w - 1) and x1 == -1:
        line_segments.append((x0, w-1))

    #print('w, line_segments ', w, line_segments)
    # for each segment, draw a rectangle, find how full that rectangle is
    density = []
    for seg in line_segments:
        x0, x1 = seg
        line_width = float(x1 - x0)
        # penalize tiny line segment, it could be spike at edge of mask
        if line_width < 5:
            line_width *= -1
        area =  np.count_nonzero(mask[cy-10:cy+10, x0:x1])
        # height of rectangle is a constant that can be ignored when compare
        density.append(area / line_width)

    #print('density ', density)
    maxd = -1
    for i,d in enumerate(density):
        if d > maxd:
            maxd = d
            cx = int(sum(line_segments[i]) / 2.0)

    #print('before shift cx cy', cx, cy)
    shift = ((-8, 0), (8, 0), (0, -8), (0, 8))
    maxd = -1
    # now (cx, cy) may still very close to border, think a V shape mask where
    # the horizontal line segment centered at the tip of V.
    # adjust cy a little bit
    if w < 40 or h < 40:
        # skip the y adjustment if mask is tiny
        pass
    elif np.count_nonzero(mask[cy-4:cy+4, cx-4:cx+4]) != 64:
        # move cx, cy around, find a point surrounded by most non-empty pixels
        for s in shift:
            x = cx + s[0]
            y = cy + s[1]
            d = np.count_nonzero(mask[y-4:y+4, x-4:x+4])
            if d > maxd:
                shift_x, shift_y = s
                maxd = d

        #print('shift_x, shift_y ', shift_x, shift_y)
        cx += shift_x
        cy += shift_y

    return (cx, cy)


def get_all_masks(db, dzscale, offset_x=None, offset_y=None, w=None, h=None):
    """ a uint16 mask, value of mask is the db id
    Args:
        dzscale: scale down factor, e.g. 2 is half the size of full resolution

    """
    if offset_x is None:
        # get canvas size to include all masks
        sql = 'select min(x), max(x+w), min(y), max(y+h) from Mask'
        row = query_db(db, sql, one=True)
        x0, x1, y0, y1 = row
        w = x1 - x0
        h = y1 - y0
        offset_x, offset_y = x0, y0

    w = int(w / dzscale) + 1
    h = int(h / dzscale) + 1
    cvsm = np.zeros((h, w), dtype=np.uint16)

    rows = query_db(db, 'select id,x,y,w,h,mask from Mask where is_bad=0')
    mids = []
    centers = {}
    orig_centers = {}
    for r in rows:
        mid,mx,my,mw,mh,blob = r
        # find real center, center of a mask can be outside, e.g. a C shape
        # mask. Draw a horizontal line, use the middle of its intersection with
        # mask as center
        mask = rleblob2mask(blob)
        #print('mid = ', mid)
        cx, cy = find_mask_center(mask, mw, mh)

        """
        # debug
        if mid == 6105:
            print('cx, cy by find_mask_center() (%d,%d), cx, cy by 0.5w (%d,%d)'
                    % (cx, cy, 0.5*mw, 0.5*mh))
            print(mw, mh)
            print('center v = %d' % mask[int(mh/2.0), int(mw/2.0+1)])
            print('center v = %d (adjusted)' % mask[cy, cx])
            exit(0)
        """

        cx += mx
        cy += my
        orig_centers[mid] = (cx, cy)

        if dzscale != 1.0:
            scaled_w = int(float(mw) / dzscale)
            scaled_h = int(float(mh) / dzscale)
            mask = utimage.resize_mask(mask, scaled_w, scaled_h)
        # new x, y, cx, cy after scale
        mx = int(float(mx - offset_x) / dzscale)
        my = int(float(my - offset_y) / dzscale)
        _put_one_mask_in_region(cvsm, mid, mx, my, mask, w, h)

        cx = int(float(cx - offset_x) / dzscale)
        cy = int(float(cy - offset_y) / dzscale)
        centers[mid] = (cx, cy)
        mids.append(mid)

    return {'cvs': cvsm, 'offset_x': offset_x, 'offset_y': offset_y,
            'dzscale': dzscale, 'mids': mids,
            'scaled_centers': centers, 'orig_centers': orig_centers }


def get_mask_neigb(con, mid, center, half_regieon=HALF_REGIEON):
    cx, cy = center
    sql = '''select id from Mask
             where cx > ? and cx < ? and
                   cy > ? and cy < ? and is_bad=0 and id <>?'''
    rows = query_db(con, sql, (cx - half_regieon, cx + half_regieon, cy - half_regieon, cy + half_regieon, mid))

    return [r[0] for r in rows]


def mask_distance_by_dilate(db, mid1, mid2):
    """ a uint16 mask, value of mask is the db id
    Args:
        dzscale: scale down factor, e.g. 2 is half the size of full resolution

    """

    k = (mid1, mid2) if mid1 < mid2 else (mid2, mid1)
    if k in dilate_cache:
        return dilate_cache[k]

    print('distance by dilate: mask %d and %d' % (mid1, mid2))
    # get canvas size to include all masks
    sql = '''select min(x), max(x+w), min(y), max(y+h) from Mask
             where id=? or id=?
    '''

    row = query_db(db, sql, (mid1, mid2), one=True)
    x0, x1, y0, y1 = row
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    masks = [np.zeros((h, w), dtype=np.uint8),
             np.zeros((h, w), dtype=np.uint8)]
    mids = (mid1, mid2)
    offset_x, offset_y = x0, y0
    sql = 'select id,x,y,w,h,mask from Mask where id=?'
    for i in range(2):
        row = query_db(db, sql, (mids[i],), one=True)
        mid,mx,my,mw,mh,blob = row
        mx -= offset_x
        my -= offset_y

        mask = rleblob2mask(blob)
        #print('mask shape ', mask.shape)
        cvs = masks[i]
        cvs[my:my+mh, mx:mx+mw][mask != 0] = 1

    dist = utimage.distance_by_dilate(masks[0], masks[1])
    print('distance by dilation = %.2f' % dist)
    dilate_cache[k] = dist
    return dist


def orig_dist(p1, p2, dzscale):
    return dzscale * math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


EQ_Y = 0
EQ_X = 1
EQ_VERTICAL = 2
def check_and_add_direct(cvs, mid1, mid2, pt1, pt2, direct_pairs, db, dzscale):
    """
    draw a line between pt1 and pt2 on cvs, check whether there is mask other
    than mid1 and mid2

    pt1 and pt2 are center of mask. In case they may be outside of mask, use
    find_mask_center() to find a alternative point

    """

    x1, y1 = pt1
    x2, y2 = pt2

    # center of mask 1 is overlap with another mask. Since center has been
    # adjusted to be inside the mask, conside distance between mask1 and mask v
    # is 0
    v = cvs[y1, x1]
    v2 = cvs[y2, x2]
    if v != 0 and v != mid1:
        print('center of mask %d is overlapping with mask %d' % (mid1, v))
        dist = 0
        _add_to_direct_pairs(direct_pairs, mid1, int(v), dist)
        return True

    #if mid2 == 6105:
    #    print('v of 6105 = ', v2)
    if v2 == 0:
        print('center of mask %d is outside' % mid2, file=sys.stderr)
        exit(1)

    # solve line equation y = ax + b
    start_checking =  False
    x, y = x1, y1
    if x1 != x2:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        if abs(a) > 2:
            # increment change y to avoid large y jump
            # solve line equation x = cy + d
            c = (x2 - x1) / (y2 - y1)
            d = x1 - c * y1
            eq_type = EQ_Y
        else:
            eq_type = EQ_X
    else:
        # line between two centers is vertical
        eq_type = EQ_VERTICAL

    pair_cord = None  # the two points used to calc distance
    # two points at the border of mask, their distance may better represent the
    # interstitue than distance of tubule centers.
    pt3, pt4 = None, None

    if eq_type == EQ_X:
        step = 1 if x2 > x1 else -1
        x_end = x2 + 1 if x2 > x1 else x2 - 1
        y = int(a * x + b)
        prev_v = cvs[y, x]

        #if mid1 == 6103 and mid2 == 6105:
        #    print('EQ_X mid1, mid2, prev_v', mid1, mid2, prev_v)

        if prev_v != mid1:
            print('Error: Center of mask1 is outside of mask')
            print('EQ_X mid1, mid2, prev_v', mid1, mid2, prev_v)
            return False
            #exit(1)

        #print('EQ_X mid1, mid2, prev_v', mid1, mid2, prev_v)
        while x != x_end:
            y = int(a * x + b)
            v = cvs[y, x]
            #print(x, x2, 'y = ', y, 'v = ', v, 'start ', start_checking)
            if prev_v == mid1 and v != mid1:
                # pt3 is at mask border
                #print('found pt3')
                pt3 = (x, y)

            #if mid1 == 6103 and mid2 == 6105:
            #    print('cur (x, y) v', x, y, v)

            if v != 0 and v != mid1:
                # reach another mask, can be mask2, or other mask sit between
                # mask1 and mask2
                pt4 = (x, y)
                if pt3 is None:
                    print('EQ_X mid1, mid2, v', mid1, mid2, v)
                    print('EQ_X pt3, pt4', pt3, pt4)
                    exit(1)

                dist = orig_dist(pt3, pt4, dzscale)
                pair_cord = (pt1, pt2, pt3, pt4)
                _add_to_direct_pairs(direct_pairs, mid1, int(v), dist,
                                     pair_cord)
                return True

            x += step
            prev_v = v

    elif eq_type == EQ_Y or eq_type == EQ_VERTICAL:
        step = 1 if y2 > y1 else -1
        y_end = y2 + 1 if y2 > y1 else y2 - 1
        if eq_type == EQ_Y:
            x = int(c * y + d)

        prev_v = cvs[y, x]
        #if mid1 == 6103 and mid2 == 6105:
        #    print('EQ_Y mid1, mid2, prev_v', mid1, mid2, prev_v)

        if prev_v != mid1:
            print('Error: Center of mask1 is outside of mask')
            print('EQ_Y mid1, mid2, prev_v', mid1, mid2, prev_v)
            return False
            #exit(1)

        while y != y_end:
            if eq_type == EQ_Y:
                x = int(c * y + d)

            v = cvs[y, x]

            #if mid1 == 9526 and mid2 == 9535:
            #    print('cur (x, y) v', x, y, v)

            if prev_v == mid1 and v != mid1:
                # pt3 is at mask border
                #if mid1 == 9526 and mid2 == 9535:
                #    print('find pt3')
                pt3 = (x, y)

            if v != 0 and v != mid1:
                # reach another mask, can be mask2, or other mask sit between
                # mask1 and mask2
                pt4 = (x, y)
                if pt3 is None:
                    print('EQ_Y mid1, mid2, v', mid1, mid2, v)
                    print('EQ_Y pt3, pt4', pt3, pt4)
                    exit(1)

                dist = orig_dist(pt3, pt4, dzscale)
                pair_cord = (pt1, pt2, pt3, pt4)
                _add_to_direct_pairs(direct_pairs, mid1, int(v), dist,
                                     pair_cord)
                return True

            y += step
            prev_v = v

    v = cvs[y, x]
    print('pt1 pt2 ', pt1, pt2)
    print('end of is_direct, something is wrong. mid1 = %d mid2 = %d cur_v = %d' %
          (mid1,mid2,v), file=sys.stderr)

    print('It could be one mask is glm and another is tubule, and they are overlap')
    dist = 0
    _add_to_direct_pairs(direct_pairs, mid1, int(v), dist)

    #_add_to_direct_pairs(direct_pairs, mid1, mid2)
    return True


def _add_to_direct_pairs(direct_pairs, mid1, mid2, dist, pair_cord=None):
    # two immediate neighboring masks
    direct_pairs['pairs'][mid1].add(mid2)
    direct_pairs['pairs'][mid2].add(mid1)

    # distance between two masks, find by walk direct line start from center of
    # one mask, or if one mask is centered inside another mask, find distance
    # by dilate one mask until reachs another mask
    tmp = (mid1, mid2) if mid1 < mid2 else (mid2, mid1)
    k = '%d_%d' % tmp
    if k not in direct_pairs['dists']:
        direct_pairs['dists'][k] = []

    if k not in direct_pairs['pair_cords']:
        direct_pairs['pair_cords'][k] = []

    if pair_cord is not None:
        direct_pairs['pair_cords'][k].append(pair_cord)

    direct_pairs['dists'][k].append(dist)


def main():
    dbf = sys.argv[1]

    con = sqlite3.connect(dbf, timeout=5)
    update_db_schema(con)
    t0 = time.time()
    dzscale = 2.0
    res = get_all_masks(con, dzscale=dzscale)

    t1 = time.time()
    # key is sorted (mid1, mid2) tuple, value is distances found by measure p3
    # p4, or found by dilate mask1 until reaches mask2
    direct_dists = {}
    pairs = {mid: set([]) for mid in res['mids']}
    direct_pairs = {'pairs': pairs, 'dists': direct_dists, 'pair_cords': {}}
    print('get_all_masks uses %.2f s' % (t1 - t0))
    for mid in res['mids']:
        center = res['orig_centers'][mid]
        pt1 = res['scaled_centers'][mid]

        nbs = get_mask_neigb(con, mid, center, half_regieon=HALF_REGIEON)
        for nb_mid in nbs:
            if (nb_mid in pairs[mid]) or (mid in pairs[nb_mid]):
                continue

            pt2 = res['scaled_centers'][nb_mid]

            """
            pts = (pt1, pt2)
            ids = (mid, nb_mid)
            for i in range(2):
                v = res['cvs'][pts[i][1], pts[i][0]]
                if v != ids[i]:
                    print('Error: center of mask %d is overlapping with mask %d' %
                            (ids[i], v))
                    print(ids)
                    exit(1)

            if mid < 6100:
                continue
            """

            check_and_add_direct(res['cvs'], mid, nb_mid, pt1, pt2,
                                 direct_pairs, con, dzscale)

        print('%d is direct sight to ' % mid, pairs[mid])

    #row = query_db(con, 'select body from Info limit 1', one=True)
    #js_body = json.loads(row[0])
    #js_body['direct_pairs'] = direct_pairs
    for mid, nbs in pairs.items():
        tmp = [x for x in nbs]
        pairs[mid] = tmp

    # ztxt1 for villin-villin, ztxt2 for glm-villin
    cur = con.cursor()
    cur.execute('update Info set ztxt1=? where title=?',
            (compress_obj(direct_pairs), 'slideinfo'))
    con.commit()
    print('db commited')


def verify_border_dist():
    """ plot a mask and border lines to  its immediate neighbors """
    import cv2
    dbf = sys.argv[1]
    con = sqlite3.connect(dbf, timeout=5)

    row = query_db(con, 'select body from Info limit 1', one=True)
    jbody = json.loads(row[0])['direct_pairs']
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


def scale_point(pt, scale):
    return (pt[0] * scale, pt[1] * scale)


def verify():
    """ plot a random mask and its immediate neighbors """
    dbf = sys.argv[1]
    con = sqlite3.connect(dbf, timeout=5)

    row = query_db(con, 'select body from Info limit 1', one=True)
    jbody = json.loads(row[0])['direct_pairs']
    direct_pairs = jbody['pairs']
    dists = jbody['dists']

    all_mids = [int(x) for x in direct_pairs]
    i = 0
    center_v = 0xffff / 6
    other_v = int(center_v / 1.5)
    res = get_all_masks(con, dzscale=2.0)
    h, w = res['cvs'].shape
    while i < 10:
        print('i = %d' % i)
        mid = random.choice(all_mids)
        cvs = np.copy(res['cvs'])
        cvs3 = np.zeros((h, w, 3), dtype=np.uint8)
        nbs = direct_pairs[str(mid)]
        print('mask %d has neighbours' % mid, nbs)

        cvs3[:, :, :][cvs != 0] = 100

        cvs3[:, :, 0][cvs == mid] = 50
        cvs3[:, :, 1][cvs == mid] = 255
        cvs3[:, :, 2][cvs == mid] = 50

        mid_dists = []
        for nb_mid in nbs:
            #cvs3[:, :, 0][cvs == nb_mid] = 255
            #cvs3[:, :, 1][cvs == nb_mid] = 0
            #cvs3[:, :, 2][cvs == nb_mid] = 255
            tmp = (mid, nb_mid) if mid < nb_mid else (nb_mid, mid)
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

        outfn = '/nfs/wsi/wei/tmp/f5_masks_%d.png' % mid
        io.imsave(outfn, cvs3)
        print('output saved to %s' % outfn)
        i += 1


def verify_nearby_masks():
    # visualize two masks to see their distance
    if len(sys.argv) != 5:
        print('usage: %s dbfile mid1 mid2' % sys.argv[0])
        return

    dbf = sys.argv[1]
    mid1 = int(sys.argv[2])
    mid2 = int(sys.argv[3])
    mid3 = int(sys.argv[4])
    con = sqlite3.connect(dbf, timeout=5)

    sql = '''select min(x), max(x+w), min(y), max(y+h) from Mask
             where id=? or id=? or id=?
    '''

    row = query_db(con, sql, (mid1, mid2, mid3), one=True)
    x0, x1, y0, y1 = row
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    cvs = np.zeros((h, w, 3), dtype=np.uint8)
    offset_x, offset_y = x0, y0
    sql = 'select id,x,y,w,h,mask from Mask where id=?'
    mids = (mid1, mid2, mid3)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        row = query_db(con, sql, (mids[i],), one=True)
        mid,mx,my,mw,mh,blob = row
        print(mid, mw, mh, mx, my)
        mx -= offset_x
        my -= offset_y

        mask = rleblob2mask(blob)

        cvs[my:my+mh, mx:mx+mw, 0][mask != 0] =  colors[i][0]
        cvs[my:my+mh, mx:mx+mw, 1][mask != 0] =  colors[i][1]
        cvs[my:my+mh, mx:mx+mw, 2][mask != 0] =  colors[i][2]

        cvs[my:my+mh, mx:mx+1] =  255
        cvs[my:my+mh, mx+mw:mx+mw+1] = 255
        cvs[my:my+1, mx:mx+mw] = 255
        cvs[my+mh:my+mh+1, mx:mx+mw] = 255

        cx, cy = find_mask_center(mask, mw, mh)
        cx += mx
        cy += my
        # mark mask center in yellow
        cvs[cy-2:cy+2, cx-2:cx+2, 0] = 255
        cvs[cy-2:cy+2, cx-2:cx+2, 1] = 255
        cvs[cy-2:cy+2, cx-2:cx+2, 2] = 0

    outfn = '/nfs/wsi/wei/tmp/verify_nearby_masks_%d_%d_%d.png' % (mid1,
            mid2, mid3)
    io.imsave(outfn, cvs)
    print('output saved to %s' % outfn)


def verify_json():
    dbf = sys.argv[1]
    con = sqlite3.connect(dbf, timeout=5)
    row = query_db(con, 'select body from Info limit 1', one=True)

    js_body = json.loads(row[0])
    dists = js_body['direct_pairs']['dists']
    for pair, v in dists.items():
        v.sort()
        diff = v[-1] - v[0]
        if diff > 100:
            print(pair, diff, v)
            print()


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
    #con_glm = sqlite3.connect(dbf, timeout=5)
    con_vil = sqlite3.connect(vil_dbf, timeout=5)

    row = query_db(con_vil, 'select body from Info limit 1', one=True)
    jbody = json.loads(row[0])['direct_pairs']
    direct_pairs = jbody['pairs']
    dists = jbody['dists']

    rows = query_db(con_vil, 'select id from Mask where is_bad=0')
    all_mids = [r[0] for r in rows]

    dzscale = 1.0

    # crop the whole slide
    sql = 'select min(x), max(x+w), min(y), max(y+h) from Mask'
    r1 = query_db(con_vil, sql, one=True)
    #r2 = query_db(con_vil, sql, one=True)
    x0, x1, y0, y1 = r1
    offset_x, offset_y = x0, y0
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    res_vil = get_all_masks(con_vil, dzscale, offset_x, offset_y, w, h)
    h, w = res_vil['cvs'].shape
    offset_x, offset_y = res_vil['offset_x'], res_vil['offset_y']

    slide = UTSlide(vil_wsi)


    # WT_Sham_9296L glm
    tgt_mids =[365,]

    # WT_D14_2755L glm
    tgt_mids = [6321,]
    tgt_mids = [5990,]

    for mid in tgt_mids:
        row = query_db(con_vil, 'select cx,cy from Mask where id=?',
                       (mid,), True)
        cx, cy = row
        # crop the canvas to rectangle centered at (cx, cy)
        cx -= offset_x
        cy -= offset_y
        hwh = 1200  # half width height

        cvs = res_vil['cvs'].copy()
        # the resulting cvs has glm mask value of mid + GLM_ID_OFFSET
        #put_glm_mask_on_villin_canvas(cvs, offset_x, offset_y, con_vil,
        #                              mid, dzscale)
        x0 = max(0, cx-hwh)
        y0 = max(0, cy-hwh)

        cvs2 = cvs[y0:cy+hwh, x0:cx+hwh]
        cvs_h, cvs_w = cvs2.shape
        # draw mask as alpha
        cvsa = np.full((cvs_h, cvs_w, 4), 0, dtype=np.uint8)

        #cvs_glm_id = mid + GLM_ID_OFFSET
        cvs_glm_id = mid
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
        outfn = '/nfs/wsi/wei/tmp/f5_vil_on_slide_%s_%d.png' % (no_ext, mid)
        io.imsave(outfn, im)
        print('output saved to %s' % outfn)



if __name__ == "__main__":
    main()
