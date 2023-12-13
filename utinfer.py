#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
from PIL import Image, ImageDraw
from utslide import UTSlide, query_db
from utslide import blob2mask, array2pngblob, blob2narray, narray2blob
from utslide import REGION_OVERLAP, MASK_OVERLAP, STEP_X, STEP_Y
from utslide import MASK_FMT_PNG, MASK_FMT_RLE
from utmask import mask2shape, mask2crop
from utmask import mask2rleblob, rleblob2mask
from utcolor import reinhard_normalization
from skimage.morphology import disk, binary_dilation
import utimage

import cv2
CV2_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

MAXINT = 9223372036854775807
#IOU_THRESHOLD = 0.95
#IOU_THRESHOLD = 0.90
IOU_THRESHOLD = 0.5
EMBED_THRESHOLD = 0.6


def get_mask_by_id(db, mid):
    sql1 = 'select id,x,y,w,h from Mask where is_bad=0'
    row = query_db(db, 'select mask from Mask where id=?', (mid,), one=True)
    return rleblob2mask(row[0])


def get_mask_rectangle(db, mid):
    sql = 'select x,y,w,h from Mask where id=?'
    r = query_db(db, sql, (mid,), one=True)
    return (r['x'], r['y'], r['w'], r['h'])


def find_perimeter_cv(mask):
    """ 0 value in mask is background """
    if CV2_MAJOR_VERSION == 4:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_TC89_KCOS)
    perimeter = 0
    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    return perimeter


def is_remove_intersection(db, m1, m2):
    """ estimate whether should remove intersection from mask1 or mask2
    Argument: m1 m2 are the id in Mask table """

    m1, m2 = int(m1), int(m2)
    print('check intersection between mask %d and %d' % (m1, m2))
    sql = 'select x,y,w,h,mask from Mask where id=?'
    r1 = query_db(db, sql, (m1,), one=True)
    r2 = query_db(db, sql, (m2,), one=True)

    x1,y1,w1,h1,blob1 = r1
    x2,y2,w2,h2,blob2 = r2

    mask1, mask2 = rleblob2mask(blob1), rleblob2mask(blob2)
    pm1 = find_perimeter_cv(mask1.copy().astype(np.uint8))
    pm2 = find_perimeter_cv(mask2.copy().astype(np.uint8))

    x, y = min(x1, x2), min(y1, y2)
    x_p, y_p = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
    w, h = x_p - x, y_p - y
    cvs1 = np.zeros((h, w), dtype=np.uint8)
    cvs2 = np.zeros((h, w), dtype=np.uint8)

    mx1, my1 = x1 - x, y1 - y
    cvs1[my1:my1+h1, mx1:mx1+w1][mask1] = 1
    cvs_intsec = cvs1.copy()

    mx2, my2 = x2 - x, y2 - y
    cvs2[my2:my2+h2, mx2:mx2+w2][mask2] = 1
    cvs_intsec[cvs2 == 0] = 0

    cvs1[my2:my2+h2, mx2:mx2+w2][mask2] = 0
    cvs2[my1:my1+h1, mx1:mx1+w1][mask1] = 0

    a1 = np.count_nonzero(mask1)
    a2 = np.count_nonzero(mask2)
    a_i = float(np.count_nonzero(cvs_intsec))  # mask1
    embed1 = a_i / a1
    embed2 = a_i / a2
    if a_i < 1000 or embed1 > EMBED_THRESHOLD or embed2 > EMBED_THRESHOLD:
        print('ignore remove intersection, a_i = %d, embed1 %.3f , embed2 %.3f '
                % (a_i, embed1, embed2))
        return None

    # perimeter of mask 1 or 2 after remove intersection
    pm1_aft = find_perimeter_cv(cvs1)
    pm2_aft = find_perimeter_cv(cvs2)

    """
         **********xxxxxxx
         *        ****   x
         *    A      * B x
         *           *   x
         *        ****   x
         *        *xxxxxxx
         **********
    A and B overlap, after remove intersection, B's perimeter increases, while
    A's perimeter decrease. Consider this intersection shall belong to B
    """

    print('mask1 perimeter = %d %d (after remove intersection)' % (pm1, pm1_aft))
    print('mask2 perimeter = %d %d (after remove intersection)' % (pm2, pm2_aft))
    if pm1_aft > pm1 and pm2_aft < pm2:
        return {'remove insec': m2, 'remove by': m1}

    if pm2_aft > pm2 and pm1_aft < pm1:
        return {'remove insec': m1, 'remove by': m2}

    return None


def overlap_calc(db, m1, m2):
    """ calculate IoU of two masks
    Argument: m1 m2 are the id in Mask table """
    print('check IoU between mask %d and %d' % (m1, m2))
    sql = 'select x,y,w,h,mask from Mask where id=?'
    r1 = query_db(db, sql, (int(m1),), one=True)
    r2 = query_db(db, sql, (int(m2),), one=True)

    x1,y1,w1,h1,blob1 = r1
    x2,y2,w2,h2,blob2 = r2

    mask1, mask2 = rleblob2mask(blob1), rleblob2mask(blob2)
    a1 = np.count_nonzero(mask1)
    a2 = np.count_nonzero(mask2)
    # make m1 the smaller mask
    if a1 == 0 or a2 ==0:
        print(a1, a2, 'empty mask')
        exit(0)

    if a2 < a1:
        tmp = m1
        m1 = m2
        m2 = tmp
        x1,y1,w1,h1,_ = r2
        x2,y2,w2,h2,_ = r1
        tmp = mask1
        mask1 = mask2
        mask2 = tmp

    x, y = min(x1, x2), min(y1, y2)
    x_p, y_p = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
    w, h = x_p - x, y_p - y
    cvs = np.zeros((h, w), dtype=np.uint8)

    x0, y0 = x1 - x, y1 - y
    cvs[y0:y0+h1, x0:x0+w1][mask1] = 1
    cvs_int = cvs.copy()

    x0, y0 = x2 - x, y2 - y
    cvs[y0:y0+h2, x0:x0+w2][mask2] = 2
    union = np.count_nonzero(cvs)

    # intersection
    area1 = np.count_nonzero(mask1)  # mask1
    cvs_int[y0:y0+h2, x0:x0+w2][mask2] = 0
    area2 = np.count_nonzero(cvs_int)  # non-overlapping part of mask1
    intersection = area1 - area2
    iou = float(intersection) / union
    embed = float(intersection) / area1
    print('IoU = %.4f, embed = %.4f' % (iou, embed))
    is_overlap = False
    if area1 == 0:  # empty mask should not exists
        is_overlap = True
        print('area1 = 0 empty mask')
        exit(0)

    elif iou > IOU_THRESHOLD:
        is_overlap = True

    elif embed > EMBED_THRESHOLD:
        # most or all mask1 is inside mask2
        is_overlap = True
        #print('float(area2) / area1) = %.4f' % (float(area2) / area1))

    return (is_overlap, iou, intersection, union)


def is_empty_region(img):
    t0 = time.time()
    im = np.asarray(img)
    if len(im.shape) != 3:
        return True

    h, w, _ = im.shape

    step = int(h / 20)
    y = 0
    while y < h:
        x = 10
        while x < w:
            r, g, b = im[y, x, :]
            if r != 255 or g != 255 or b != 255:
                print('detect empty region uses %.3s' % (time.time() - t0))
                return False
            x += step

        y += step

    print('detect empty region uses %.3s' % (time.time() - t0))
    return True


def delete_tiny_mask(slide=None, slide_path=None, min_size=10):
    """ min_size in um"""

    if slide is None:
        slide = UTSlide(slide_path)

    # delete mask smaller than 10um
    min_w = min_h = min_size / slide.mpp
    #print(min_w,min_h)
    q = query_db(slide.db,
            'select count(*) from Mask where w<? or h<?', (min_w, min_h), True)
    print('total %d tiny masks' % q[0])
    slide.db.execute('delete from Mask where w<? or h<?', (min_w, min_h))
    slide.db.commit()


def merge_overlap_mask_v2(slide=None, slide_path=None, mode='iou'):
    """ merge mask one by one, new Zeiss creates image too big to fit into
    RAM """

    print('Checking overlapping by %s' % mode)
    if slide is None:
        slide = UTSlide(slide_path)

    cur = slide.db.execute('pragma journal_mode=wal;')
    row = cur.fetchone()
    print('result of switch pragma journal_mode to wal = ', row[0])

    sql = 'select id from Mask where is_bad=0'
    rows = query_db(slide.db, sql)
    chains = {}
    standalong_ids = set([])
    de_overlap = []
    for r in rows:
        overlapping_ids, tmp = get_overlapping_masks(slide.db, r['id'],
                                        mode, standalong_ids)
        de_overlap.extend(tmp)
        if not overlapping_ids:
            standalong_ids.add(r['id'])
            continue

        # chains: {m1: {m1, m2...}, m2: {m2, m1...}, m4: {m5, m6...}...}
        for mid in overlapping_ids:
            if mid not in chains:
                chains[mid] = set([])

            for i in overlapping_ids:
                chains[mid].add(i)

    dedup = set([])
    merge_chains_v2(chains, dedup)

    # insert/update many masks is I/O intensive, wrap them in a single
    # transaction.
    t0 = time.time()
    slide.db.isolation_level = None
    cur = slide.db.cursor()
    cur.execute('BEGIN')

    # remove intersection from one of the overlapping mask, keep the mask id
    for pair in de_overlap:
        remove_intersection(slide.db, cur, from_mid=pair['remove insec'],
                                 by_mid=pair['remove by'])
    #if de_overlap:
    #    cur.execute('COMMIT')

    for mid, chain in chains.items():
        if mid in dedup:
            continue

        print('merging ', chain)
        merge_masks(slide.db, cur, chain)

    cur.execute('COMMIT')
    slide.db.isolation_level = ''
    cur = slide.db.execute('pragma journal_mode=delete;')
    row = cur.fetchone()
    slide.db.close()


def find_perimeter(mask):
    """
    Args:
        mask: 2D array, 0 for background, a non-zero value other than 255 for
              foreground
    """

    # highlight the border pixel in 255
    utimage.highlight_mask_border(mask)
    uniq, counts = np.unique(mask, return_counts=True)
    for i,v in enumerate(uniq):
        if v == 255:
            return counts[i]

    return -1


STRAIGHT_DELTA = 5 # max dx or dy for horizontal or vertical line
STRAIGHT_LEN = 50  # straight line longer than 50
def calc_roundness(mask):
    """ area / perimeter
    circle 0.282
    square 0.25
    rectangle:
        2x1: 0.2357    3x1: 0.2165    4x1: 0.2
        5x1: 0.1863    9x1: 0.15     12x1: 0.1332    14x1: 0.1247
    """

    area = np.count_nonzero(mask)
    if CV2_MAJOR_VERSION == 4:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_TC89_KCOS)
    perimeter = 0
    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    # looking for long straight vertical or horizotal line, these masks are at
    # the border of sliding regions. Intersection has this kind of look shall
    # be merged together
    shapes = []
    for cnt in contours:
        total = cnt.shape[0]
        points = np.reshape(cnt, (total, 2))
        count = len(points)
        shapes.append((count, points))

    shapes.sort(key=lambda x: x[0])  # choose the shape with most edge points
    s = [p for p in shapes[-1][1]]
    s.append(s[0])
    npoints = len(s)
    i = 0
    while i < (npoints - 1):
        p1, p2 = s[i], s[i+1]

        dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
        if dx < STRAIGHT_DELTA and dy > STRAIGHT_LEN:
            # use 1.0 to indicate two masks shall be merged
            print('straight line', dx, dy)
            return 1.0

        elif dy < STRAIGHT_DELTA and dx > STRAIGHT_LEN:
            print('straight line', dx, dy)
            return 1.0

        i += 1

    print('perimeter = ', perimeter)
    return np.sqrt(area) / perimeter


def get_overlapping_masks(db, mid, mode, known_standalongs):
    """ all masks overlap with this rectangle area """

    x, y, w, h = get_mask_rectangle(db, mid)
    sql = ('SELECT id,x,y,w,h,mask FROM Mask '
           ' WHERE ((x+w>=? AND x+w<=?) OR (x>=? AND x<=?) '
           '        OR (x<=? AND x+w>=?) OR (x<=? AND x+w>=?)) '
           '   AND ((y+h>=? AND y+h<=?) OR (y>=? AND y<=?) '
           '        OR (y<=? AND y+h>=?) OR (y<=? AND y+h>=?)) '
           '   AND is_bad=0 '
           ' ORDER BY id ')

    x1, y1 = x + w, y + h
    rows = query_db(db, sql, (x, x1, x, x1, x, x, x1, x1,
                              y, y1, y, y1, y, y, y1, y1))
    masks = {}
    overlapping_ids = []
    de_overlap = []

    if len(rows) == 1:
        return (overlapping_ids, de_overlap)

    x1, y1 = MAXINT, MAXINT
    x2, y2 = 0, 0
    for r in rows:
        if r['id'] in known_standalongs:
            continue

        m = {'id': r['id'], 'x': r['x'], 'y': r['y'],
             'w': r['w'], 'h': r['h'], 'mask': rleblob2mask(r['mask'])}
        masks[r['id']] = m
        x1 = min(x1, r['x'])
        y1 = min(y1, r['y'])
        x2 = max(x2, r['x'] + r['w'])
        y2 = max(y2, r['y'] + r['h'])

    w = x2 - x1
    h = y2 - y1
    cvs = np.zeros((h, w), dtype=np.uint16)
    for _, m in masks.items():
        mx = m['x'] - x1
        my = m['y'] - y1
        cvs[my: my + m['h'], mx: mx + m['w']][m['mask']] = m['id']

    m0 = masks[mid]
    mx = m0['x'] - x1
    my = m0['y'] - y1

    # copy mask region from the working canvas for dup checking, if there
    # is overlap, the copied region will have many nonzero values, each
    # nonzero value is the sqlite id of a unique mask
    cvstmp = cvs[my: my + m0['h'], mx: mx + m0['w']].copy()
    cvstmp[m0['mask'] == 0] = 0
    # later check roundness of an overlapping area
    h2, w2 = cvstmp.shape
    uniq, overlap = np.unique(cvstmp, return_counts=True)
    # it is also possible that mid is entirely covered by other masks, then
    # uniq[1] will be that overlapping mask
    if len(uniq) == 2 and int(uniq[1]) == mid:
        # this mask does not overlap with other masks
        return (overlapping_ids, de_overlap)

    for i in range(len(uniq)):
        if uniq[i] == 0:  # background
            continue

        if uniq[i] == mid:
            continue

        cvs_mask_id = uniq[i]
        if mode == 'iou':
            is_overlap, iou, _, _ = overlap_calc(db, mid, cvs_mask_id)
            #print('IoU = %.4f' % iou)
            if not is_overlap:
                continue

        elif mode == 'area':
            # long tubule sometimes has small thin piece over hanging, which
            # overlapping with neighboring tubule. It is seeing on PointRend
            # masks. Try use shape feature to exclude them. Only consider fat
            # overlapping area as true overlap.
            area = overlap[i]
            if area < 500:
                continue

            # check roundness of overlap
            cvs2 = np.zeros((h2, w2), dtype=np.uint8)
            cvs2[cvstmp == cvs_mask_id] = 1
            roundness = calc_roundness(cvs2)
            print('roundness = %.4f' % roundness)
            # ignore small and thin overlap. 0.13 is too thin. 0.15 works well
            if roundness < 0.155:
                # TODO, erode the intersection from larger mask, if
                # - intersection > 4000 pixels
                res = is_remove_intersection(db, mid, cvs_mask_id)
                if res is not None:
                    de_overlap.append(res)

                continue

        overlapping_ids.append(cvs_mask_id)

    if overlapping_ids:
        overlapping_ids.append(mid)

    return ([int(x) for x in overlapping_ids], de_overlap)


def merge_chains(chains, dedup):
    """ merge chain will likely make more chains share common mask ids, need
    recursive call """
    for mid, chain in chains.items():
        if mid in dedup:
            continue

        tmp = chain.copy()
        # merge mid's neighbor
        for ovp_m in tmp:
            # and neighbor's neighbor
            for mid2 in chains[ovp_m]:
                chain.add(mid2)

            # chains[ovp_m] has been merged into chain, next time skip ovp_m
            if ovp_m != mid:
                dedup.add(ovp_m)

        if len(tmp) != len(chain):
            merge_chains(chains, dedup)


def merge_chains_v2(chains, dedup):
    """ merge chain will likely make more chains share common mask ids, need
    recursive call """
    for mid, chain in chains.items():
        if mid in dedup:
            continue

        merge_chain_one_mask(chains, dedup, mid)


def merge_chain_one_mask(chains, dedup, mid):
    if mid in dedup:
        return

    chain = chains[mid]
    tmp = chain.copy()
    for ovp_m in tmp:
        for mid2 in chains[ovp_m]:
            chain.add(mid2)

        # chains[ovp_m] has been merged into chain, next time skip ovp_m
        if ovp_m != mid:
            dedup.add(ovp_m)

    # keep merge until no new mid can be added to chain
    if len(tmp) != len(chain):
        merge_chain_one_mask(chains, dedup, mid)


def remove_intersection(db, cur, from_mid, by_mid):
    print('Remove %d intersection from %d' % (by_mid, from_mid))
    sql = 'select x,y,w,h,mask from Mask where id=?'
    r1 = query_db(db, sql, (from_mid,), one=True)
    r2 = query_db(db, sql, (by_mid,), one=True)

    x1,y1,w1,h1,blob1 = r1
    x2,y2,w2,h2,blob2 = r2
    mask1, mask2 = rleblob2mask(blob1), rleblob2mask(blob2)

    x, y = min(x1, x2), min(y1, y2)
    x_p, y_p = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
    w, h = x_p - x, y_p - y
    cvs1 = np.zeros((h, w), dtype=bool)
    cvs2 = np.zeros((h, w), dtype=bool)

    mx1, my1 = x1 - x, y1 - y
    cvs1[my1:my1+h1, mx1:mx1+w1][mask1] = 1

    mx2, my2 = x2 - x, y2 - y
    cvs2[my2:my2+h2, mx2:mx2+w2][mask2] = 1

    # dilate mask2 a little bit, then use it to remove intesection
    cvs2_dilated = binary_dilation(cvs2, disk(4))
    cvs1[cvs2] = 0

    crop = mask2crop(cvs1)
    if not crop:
        print('debug: cvs1 nonzeros = ', np.count_nonzero(cvs1))

    x += crop['x']
    y += crop['y']
    cx = x + int(crop['w'] * 0.5)
    cy = y + int(crop['h'] * 0.5)
    sql = ('UPDATE Mask set x=?,y=?,w=?,h=?,cx=?,cy=?,'
           'mask=?,mask_area=?,polygon=? WHERE id=?')
    cur.execute(sql, (x, y, crop['w'], crop['h'],cx, cy,
                      crop['blob'], crop['area'], None, from_mid))
    # db commit by caller


def merge_masks(db, cur, mask_ids):
    """ merge masks into a new mask, and mark old masks as bad """

    sql = 'select x,y,w,h,mask from Mask where id=?'
    x1, y1 = MAXINT, MAXINT
    x2, y2 = 0, 0
    masks = []
    for mid in mask_ids:
        r = query_db(db, sql, (int(mid),), one=True)
        m = {'x': r['x'], 'y': r['y'],
             'w': r['w'], 'h': r['h'], 'mask': rleblob2mask(r['mask'])}
        masks.append(m)
        x1 = min(x1, r['x'])
        y1 = min(y1, r['y'])
        x2 = max(x2, r['x'] + r['w'])
        y2 = max(y2, r['y'] + r['h'])

    w = x2 - x1
    h = y2 - y1
    cvs = np.zeros((h, w), dtype=bool)
    for m in masks:
        mx = m['x'] - x1
        my = m['y'] - y1
        cvs[my: my + m['h'], mx: mx + m['w']][m['mask']] = 1

    sql2 = 'INSERT INTO Mask (x,y,w,h,mask,mask_area) VALUES(?,?,?,?,?,?)'
    #cvstmp = np.array(cvstmp, dtype=bool)
    area = np.count_nonzero(cvs)
    blob = mask2rleblob(cvs)
    arg = (int(x1), int(y1), int(w), int(h), blob, area)
    cur.execute(sql2, arg)

    for mid in mask_ids:
        #cur.execute('update Mask set is_bad=1 where id=?', (mid,))
        cur.execute('delete from Mask where id=?', (mid,))


def remove_island_from_all_masks(slide=None, slide_path=None):
    """ remove island from mask if there is overlapping """

    if slide is None:
        slide = UTSlide(slide_path)

    cur = slide.db.execute('pragma journal_mode=wal;')
    row = cur.fetchone()
    print('result of switch pragma journal_mode to wal = ', row[0])

    rows = query_db(slide.db, 'select id from Mask')
    mask_ids = [r[0] for r in rows]
    remove_mask_island(slide, mask_ids)

    cur = slide.db.execute('pragma journal_mode=delete;')
    row = cur.fetchone()
    print('result of switch pragma journal_mode to delete = ', row[0])
    slide.db.close()


def update_db_schema(db):
    # update Mask table, add polygon column
    sql = ("SELECT count(*) FROM pragma_table_info('Mask') "
           " WHERE name='polygon'")
    row = query_db(db, sql, one=True)
    if row and row[0] == 0:
        db.execute('ALTER TABLE Mask ADD COLUMN polygon BLOB;')
        db.commit()

    sql_add_table = """
CREATE TABLE IF NOT EXISTS SlideTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile BLOB
);

CREATE TABLE IF NOT EXISTS MaskTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile BLOB,
    fullsize_mask BLOB
);

"""
    db.executescript(sql_add_table)
    db.commit()


def remove_mask_island(slide, mask_ids):
    """ remove overhanging island from masks
    some masks have small pieces of island outside of main mask area """

    print('checking and removing island for %d masks' % len(mask_ids))
    update_db_schema(slide.db)

    sql = 'select x,y,w,h,mask,polygon from Mask where id=?'
    sql_update = 'update Mask set x=?,y=?,w=?,h=?,mask=?,polygon=?,mask_area=? where id=?'

    padding = 100
    for mid in mask_ids:
        x,y,w,h,blobm,polygon = query_db(slide.db, sql, (int(mid),), one=True)
        if polygon:
            continue

        mask = rleblob2mask(blobm)
        cvs = np.zeros((h+padding*2, w+padding*2), dtype=np.uint8)
        cvs[padding:padding+h, padding:padding+w] = mask
        points = mask2shape(cvs, slide.mpp)
        xmin, ymin = [int(x) for x in points.min(axis=0)]
        xmax, ymax = [int(x) for x in points.max(axis=0)]
        w0, h0 = xmax - xmin, ymax - ymin
        if w0 == 0 or h0 == 0:
            print('error: shape w = %d, h = %d' % (w, h))
            continue

        x0 = x - padding + xmin
        y0 = y - padding + ymin
        im = np.zeros((h+padding*2, w+padding*2), dtype=np.uint8)
        img = Image.fromarray(im)
        draw = ImageDraw.Draw(img)
        points = [tuple(p) for p in points]
        draw.polygon(xy=points, fill=1)
        im = np.asarray(img)
        mask = im[ymin:ymax, xmin:xmax].copy()
        h1, w1 = mask.shape
        if h1 == 0 or w1 ==0:
            print('error: shape w = %d, h = %d' % (w, h))
            print(ymin, ymax, xmin, xmax)
            print(mask.shape, im.shape)

        blobm = mask2rleblob(mask)

        points = [[p[1]-padding, p[0] - padding] for p in points]
        points = np.asarray(points)
        blobp = narray2blob(points)
        mask_area = np.count_nonzero(mask)
        slide.db.execute(sql_update, (x0,y0,w0,h0,blobm,blobp,mask_area,mid))

    slide.db.commit()


def utround(f):
    """ round a positive number """
    return int(f + 0.5)

