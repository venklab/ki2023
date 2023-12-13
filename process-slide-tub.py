#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from skimage import io
from skimage.filters import gaussian

import matplotlib.pyplot as plt

from utslide import UTSlide
from utslide import query_db
from utslide import upgradedb1_mask_add_note
from utslide import upgradedb2_mask_add_mm_seg
from utslide import upgradedb3_mask_add_mm_pas
from utslide import upgradedb4_mask_add_is_bad
from utslide import upgradedb5_mask_add_nuclei_and_opened
from utslide import blob2mask, array2pngblob, blob2narray, narray2blob
from utslide import REGION_OVERLAP, MASK_OVERLAP, STEP_X, STEP_Y
from utslide import MASK_FMT_PNG, MASK_FMT_RLE
from utmask import masklocation, mask2shape, mask2crop
from utmask import mask2rleblob, rleblob2mask
from utinfer import delete_tiny_mask
from utinfer import merge_overlap_mask_v2
from utinfer import remove_island_from_all_masks
from utinfer import utround


APPDIR = os.path.abspath(os.path.dirname(__file__))

PREDICT_THRESH = 0.20  # villin 2023-10-06
IOU_THRESHOLD = 0.95

from utmodel import get_prediction
from utmodel import ut_load_pr_model
from utmodel import get_fb_prediction, im_get_fb_prediction
import utimage

model = None


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


def img_gamma_correction(img, gamma):
    im = np.asarray(img)
    return utimage.gamma_correct(im, gamma)


def process_slide(slide_path, model_weight=saved_fb_model, x_offset=0,
                  y_offset=0):
    slide = UTSlide(slide_path)
    slide.x_offset = x_offset
    slide.y_offset = y_offset
    model = ut_load_pr_model(model_weight)
    print('loading model %s' % model_weight)
    print('model loaded')

    commit_counter = 0
    t0 = time.time()
    border_limit = int(slide.region_overlap / 2 - slide.mask_overlap)
    j = 0
    while True:
        t0 = time.time()
        img = slide.next2()  # next image is overlapping with previous one
        if img is None:
            print('region is empty')
            continue

        if img.size == 0:
            break

        #masks = get_prediction(model, img, PREDICT_THRESH)
        masks, scores = get_fb_prediction(model, img)
        #im = img_gamma_correction(img, 0.45)
        sql = 'INSERT INTO Mask (x,y,w,h,mask,pred_score) VALUES(?,?,?,?,?,?)'
        # store masks here before insert into db, need remove duplicated mask
        mask_db_buf = []
        for i, mask in enumerate(masks):
            pred_score = scores[i]
            if pred_score < PREDICT_THRESH:
                continue

            loc = masklocation(mask)
            if not loc:
                continue

            offset_x, offset_y = slide.x, slide.y
            # 'dist': (ymin, w - xmax, h - ymax, xmin)}
            top, right, bottom, left = loc['dist']
            #print(loc['bbox'], loc['dist'])
            if top < border_limit:
                mask = mask[border_limit:, :].copy()
                offset_y += border_limit
            if bottom < border_limit:
                mask = mask[:slide.step_y - border_limit, :].copy()
            if right < border_limit:
                mask = mask[:, :slide.step_x - border_limit].copy()
            if left < border_limit:
                mask = mask[:, border_limit:].copy()
                offset_x += border_limit

            crop = mask2crop(mask)
            if not crop:
                continue

            x = offset_x + crop['x']
            y = offset_y + crop['y']
            cx = x + crop['w'] * 0.5
            cy = y + crop['h'] * 0.5
            arg = (x, y, crop, cx, cy, pred_score)

            arg = (x, y, crop, cx, cy, pred_score)
            mask_db_buf.append(arg)

        if len(mask_db_buf) == 0:
            continue

        for m in mask_db_buf:
            x, y, crop, _, _, pred_score = m

            slide.cur.execute(sql,
                (x, y, crop['w'], crop['h'], crop['blob'], pred_score))

            print('    new mask at (%d,%d), w=%d, h=%d, pred_score=%.3f' %
                    (x, y, crop['w'], crop['h'], pred_score))
            commit_counter += 1
            if commit_counter % 16 == 0:
                slide.db.commit()

    slide.db.commit()
    return


def generate_mask_tile(slide=None, slide_path=None, force=False, color=True):
    if slide is None:
        slide = UTSlide(slide_path)

    # recursively generate tile up to the top level
    if force:
        slide.db.execute('DELETE FROM Tile')
        slide.db.commit()

    slide.init_mask_tile()
    lock = None  # slide.get_mask_tile expects a database write lock
    slide.get_mask_tile(slide.db, lock, 0, (0,0), color)
    slide.db.commit()


def mark_small_glm_mask(slide=None, slide_path=None):
    if slide is None:
        slide = UTSlide(slide_path)

    sql = 'SELECT id,w,h FROM Mask WHERE is_bad=0'
    sql_mark_bad = 'UPDATE Mask SET is_bad=1 where id=?'

    rows = query_db(slide.db, sql)
    count = 0
    total = len(rows)
    MIN_GOOD = 20  # conside mask smaller than 60x60um bad (10x10 nuclei)
    min_w = utround(MIN_GOOD / slide.mpp)
    min_h = min_w
    print('mark_small_glm_mask, set min_w = ', min_w)
    for r in rows:
        if r['w'] > min_w and r['h'] > min_h:
            continue

        slide.db.execute(sql_mark_bad, (r['id'],))
        count += 1

    slide.db.commit()
    print("marked %d small glm as bad" % count)


def update_slide_info(slide=None, slide_path=None, model_file=None):
    if slide is None:
        slide = UTSlide(slide_path)

    row_info = query_db(slide.db, 'select id,body from Info', one=True)

    res = {}
    if row_info:
        row_id = row_info['id']
        res = json.loads(row_info['body'])

    if not res:
        res = {'count': 0, 'mme_mean': 0, 'mme_std': 0}
        res['dim'] = '%dx%d' % (slide.width, slide.height)
        if model_file:
            res['model_ver'] = os.path.splitext(os.path.basename(model_file))[0]

    res['time'] = time.time()

    if row_info:
        sql = 'UPDATE Info set body=? WHERE id=?'
        slide.db.execute(sql, (json.dumps(res), row_id))
    else:
        sql = 'INSERT INTO Info (title,body,maskfmt) VALUES(?,?,?)'
        slide.db.execute(sql, ('slideinfo', json.dumps(res), MASK_FMT_RLE))

    # make sure not in WAL mode
    slide.db.execute('pragma journal_mode=delete;')
    slide.db.commit()
    #slide.db.execute('VACUUM')


def update_slide_info_aftermerge(slide=None, slide_path=None):
    if slide is None:
        slide = UTSlide(slide_path)

    row = query_db(slide.db, 'select count(*) from Mask', one=True)
    count = int(row[0])

    row = query_db(slide.db, 'select id,body from Info', one=True)
    row_id = row['id']
    res = json.loads(row['body'])
    res['count'] = count

    sql = 'UPDATE Info set body=? WHERE id=?'
    slide.db.execute(sql, (json.dumps(res), row_id))
    slide.db.commit()
    slide.db.execute('VACUUM')


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', '--model-weight', help='saved model weight',
                        type=str)
    parser.add_argument("slide_path", help="path to slide file", type=str)
    parser.add_argument('-i', '--inference', action='store_true',
                        help='need inference')
    parser.add_argument('-m', '--merge', action='store_true',
                        help='need merge')
    parser.add_argument('-t', '--tile', action='store_true', help='need tile')
    args = parser.parse_args()
    need_inference = args.inference
    need_merge = args.merge
    need_mask_tile = args.tile
    slide_path = args.slide_path
    print(args)
    if need_inference:
        process_slide(slide_path, args.model_weight)
        # go through slide twice, second time shift the reading window a little
        # bit, sometimes it can pickup missing object.
        process_slide(slide_path, args.model_weight,
                      x_offset=100, y_offset=100)
        update_slide_info(None, slide_path, args.model_weight)
        bak_db(slide_path, 'unmerged')

    if need_merge:
        remove_island_from_all_masks(None, slide_path)
        merge_overlap_mask_v2(None, slide_path, mode='iou')
        merge_overlap_mask_v2(None, slide_path, mode='area')
        delete_tiny_mask(None, slide_path, 10)
        merge_overlap_mask_v2(None, slide_path, mode='iou')
        update_slide_info_aftermerge(None, slide_path)

    #mark_small_glm_mask(None, slide_path)

    if need_mask_tile:
        generate_mask_tile(None, slide_path, force=True, color=False)
        #generate_mask_tile(None, slide_path, force=True, color=True)

    #update_mme_score_recalc(None, slide_path)
    update_slide_info(None, slide_path, args.model_weight)


if __name__ == "__main__":
    main()
