#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import os
import sqlite3
import sys
import time
import zlib

import numpy as np

MPP = 0.5
# cortex thickness around 2mm, get mask in a rectangle 2mm x 2mm
# 2mm may cross to other side of kidney, reduce to 1.4mm
HALF_REGIEON = int(700 / MPP)

# make sure following is line 20
parm_n = 1

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

dir_glm = 'diretory holds slide file and glomeruli segmentation sqlite file'
dir_villin = 'diretory holds slide file and villin segmentation sqlite file'


PT_CACHE = {}
EXCLUDE_MICE = set(['2757', '2786', '9329'])
GLM_ID_OFFSET = 30000


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


def extract_info(fname):
    """ extract group, mouse tag, antibody from filename """
    lfn = fname.lower()
    if 'villin' in lfn:
        ab = 'Villin'
    elif 'krt20' in lfn:
        ab = 'Krt20'
    else:
        ab = 'Unknown'
    tmp = os.path.basename(fname).split('_')
    grp = '_'.join(tmp[:2])
    tag = tmp[2]
    if tag[-1] == 'R':
        # Use right kidney as Sham
        grp = grp.replace('D14', 'Sham')

    return { 'grp': grp, 'tag': tag, 'ab': ab, }


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


def update_db_schema(db):
    # update Mask table, add several real value column
    # r1 closest villin to villin
    # r2 closest villin to glm
    # r3 furthest glm-villin distance
    # r4 furthest villin-villin distance

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


def get_mask_center_one_slide(dbfile):
    con = sqlite3.connect(dbfile, timeout=5)
    #sql = 'select x+w*0.5 as cx, y+h*0.5 as cy, id from Mask where is_bad=0'
    sql = 'select cx,cy,id from Mask where is_bad=0'
    rows = query_db(con, sql)
    points = {r[2]: (r[0], r[1]) for r in rows}
    slide_name = dbfile.split('/')[-1][:-3]
    res = {'name': slide_name,
           'points': points,
           }
    #info = extract_info(dbfile)
    #res.update(info)
    #print(info, slide_name)
    return res


def get_masks_in_regieon(con, mid, pt, half_regieon=HALF_REGIEON):
    sql = '''select cx,cy,id from Mask
             where cx > ? and cx < ? and
                   cy > ? and cy < ? and is_bad=0 and id <>?'''
    cx, cy = pt
    rows = query_db(con, sql, (cx - half_regieon, cx + half_regieon, cy - half_regieon, cy + half_regieon, mid))
    points = [(r[0], r[1]) for r in rows]

    return points


def calc_vil_closest_N(con, mid, pt, dbf, nclosest=50):
    #min_d = MAXINT
    dists = []
    #t0 = time.time()
    ngb_points = get_masks_in_regieon(con, mid, pt)
    for p in ngb_points:
        d = (pt[0] - p[0]) ** 2 + (pt[1] - p[1]) ** 2
        dists.append(d)

    dists.sort()
    if not dists:
        return 0

    return float(np.mean( np.sqrt(np.array(dists[:nclosest])) ))


def update_villin_closest(villin_dbf, nclosest=50):
    # calc shortest distance to villin for each glm, save to db
    con = sqlite3.connect(villin_dbf, timeout=5)
    update_db_schema(con)

    vil = get_mask_center_one_slide(villin_dbf)
    args = []
    t0 = time.time()
    for mid, pt in vil['points'].items():
        shortest_dist = calc_vil_closest_N(con, mid, pt, villin_dbf, nclosest)
        args.append((shortest_dist, mid))

    #print('calcuate all distance uses %.2f seconds' % (time.time() - t0))
    con.isolation_level = None
    cur = con.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute('update Mask set r1=? where id=?', arg)

    con.isolation_level = ''
    cur.execute('COMMIT')


def update_glm_villin_closest(glm_dbf, nclosest=50):
    """
    calc average of distance of closest 50 villin, save to Mask->r1 column
    """
    dirname = os.path.dirname(glm_dbf)
    bname = os.path.basename(glm_dbf)
    vil_dbf = os.path.abspath(os.path.join(dirname, '../villin', bname))
    glm_dbf = os.path.abspath(os.path.join(dirname, '../glm', bname))

    con_glm = sqlite3.connect(glm_dbf, timeout=5)
    con_vil = sqlite3.connect(vil_dbf, timeout=5)
    update_db_schema(con_glm)
    update_db_schema(con_vil)

    glm = get_mask_center_one_slide(glm_dbf)
    args = []
    t0 = time.time()
    for mid, pt in glm['points'].items():
        shortest_dist = calc_vil_closest_N(con_vil, mid+GLM_ID_OFFSET,
                                           pt, vil_dbf, nclosest)
        args.append((shortest_dist, mid))

    con_glm.isolation_level = None
    cur = con_glm.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute('update Mask set r2=? where id=?', arg)

    con_glm.isolation_level = ''
    cur.execute('COMMIT')


def square_dist(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2


def get_pt_by_mid(con, mid):
    if mid in PT_CACHE:
        return PT_CACHE[mid]

    row = query_db(con, 'select cx, cy from Mask where id=?', (mid,), one=True)
    pt = (row[0], row[1])
    PT_CACHE[mid] = pt
    return pt


def retired_update_villin_to_villin_avg(villin_dbf):
    # calc shortest distance to villin for each glm, save to db

    t0 = time.time()
    args = []
    con = sqlite3.connect(villin_dbf, timeout=5)
    row = query_db(con, 'select body from Info limit 1', one=True)
    direct_pairs = json.loads(row[0])['direct_pairs']
    all_mids = [int(x) for x in direct_pairs]
    for mid in all_mids:
        pt1 = get_pt_by_mid(con, mid)
        dists = []
        for nb_mid in direct_pairs[str(mid)]:
            pt2 = get_pt_by_mid(con, nb_mid)
            dists.append(square_dist(pt1, pt2))

        dists.sort(reverse=True)
        avg = np.mean(np.sqrt(dists[:5]))
        args.append((avg, mid))

    print('calcuate all distance uses %.2f seconds' % (time.time() - t0))
    con.isolation_level = None
    cur = con.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute('update Mask set r3=? where id=?', arg)

    con.isolation_level = ''
    cur.execute('COMMIT')


def update_x_to_villin_avg_f5(dbf, mtype='tubule', num=5):
    # calc avg of furthest 5 villin for each villin, save to db
    # the distance is from border to border instead of center to center

    dirname = os.path.dirname(dbf)
    bname = os.path.basename(dbf)
    vil_dbf = os.path.abspath(os.path.join(dirname, '../villin', bname))
    glm_dbf = os.path.abspath(os.path.join(dirname, '../glm', bname))
    if mtype == 'glm':
        con = sqlite3.connect(glm_dbf, timeout=5)
        sql = 'select ztxt2 from Info limit 1'
    else:
        con = sqlite3.connect(vil_dbf, timeout=5)
        sql = 'select ztxt1 from Info limit 1'

    args = []

    row = query_db(con, sql, one=True)
    js_body = uncompress_obj(row[0])
    direct_pairs = js_body['pairs']
    pair_dists = js_body['dists']
    for pair, v in pair_dists.items():
        v.sort()

    rows = query_db(con, 'select id from Mask where is_bad=0')
    all_mids = [int(r[0]) for r in rows]
    for mid in all_mids:
        if mtype == 'glm':
            mid += GLM_ID_OFFSET

        dists = []
        for nb_mid in direct_pairs[str(mid)]:
            k = (mid, nb_mid) if mid < nb_mid else (nb_mid, mid)
            strk = '%d_%d' % k
            # use the longest unblocked distance between two masks
            dists.append(pair_dists[strk][-1])

        dists.sort(reverse=True)
        if not dists:
            print('not dists for mask %d' % mid)

        avg = np.mean(dists[:num])
        if mtype == 'glm':
            mid -= GLM_ID_OFFSET
        args.append((avg, mid))

    if mtype == 'glm':
        sql_upd = 'update Mask set r3=? where id=?'
    else:
        sql_upd = 'update Mask set r4=? where id=?'

    con.isolation_level = None
    cur = con.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute(sql_upd, arg)

    con.isolation_level = ''
    cur.execute('COMMIT')


def retired_update_villin_to_villin_avg_v4(villin_dbf):
    # calc shortest distance to villin for each villin, save to db
    # the distance is from border to border instead of center to center
    args = []
    con = sqlite3.connect(villin_dbf, timeout=5)
    row = query_db(con, 'select body from Info limit 1', one=True)
    js_body = json.loads(row[0])['direct_pairs']
    direct_pairs = js_body['pairs']
    pair_dists = js_body['dists']
    for pair, v in pair_dists.items():
        v.sort()

    all_mids = [int(x) for x in direct_pairs]
    for mid in all_mids:
        dists = []
        for nb_mid in direct_pairs[str(mid)]:
            k = (mid, nb_mid) if mid < nb_mid else (nb_mid, mid)
            strk = '%d_%d' % k
            # use the longest unblocked distance between two masks
            dists.append(pair_dists[strk][-1])

        dists.sort(reverse=True)
        avg = np.mean(dists[:5])
        args.append((avg, mid))

    con.isolation_level = None
    cur = con.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute('update Mask set r4=? where id=?', arg)

    con.isolation_level = ''
    cur.execute('COMMIT')


MAXINT = 1 << 63

def calc_shortest(pt, points):
    #min_d = MAXINT
    dists = []
    for p in points:
        d = (pt[0] - p[0]) ** 2 + (pt[1] - p[1]) ** 2
        dists.append(d)

    dists.sort()
    # 5 shortest
    return float(np.mean( np.sqrt(np.array(dists[:5])) ))


def update_glm_closest_one(villin_dbf):
    # calc shortest distance to villin for each glm, save to db
    bname = os.path.basename(villin_dbf)
    glm_dbf = os.path.join(dir_glm, bname)
    print('use glm_dbf %s' % glm_dbf)

    glm = get_mask_center_one_slide(glm_dbf)
    vil = get_mask_center_one_slide(villin_dbf)
    args = []
    t0 = time.time()
    for mid, pt in glm['points'].items():
        vil_points = [p for p in vil['points'].values()]
        shortest_dist = calc_shortest(pt, vil_points)
        args.append((shortest_dist, mid))

    #print('calcuate all distance uses %.2f seconds' % (time.time() - t0))
    con = sqlite3.connect(glm_dbf, timeout=5)
    update_db_schema(con)
    con.isolation_level = None
    cur = con.cursor()
    cur.execute('BEGIN')
    for arg in args:
        cur.execute('update Mask set r2=? where id=?', arg)

    con.isolation_level = ''
    cur.execute('COMMIT')


def main():
    if len(sys.argv) == 1:
        print('Usage: %s slide_db' % sys.argv[0])
        return

    dbf = sys.argv[1]

    db = sqlite3.connect(dbf)
    cur = db.execute('pragma journal_mode=wal;')
    row = cur.fetchone()
    print('result of switch pragma journal_mode to wal = ', row[0])

    parm_n = 50
    # r1 closest villin to villin
    if 1:
        print('Update closest 50 inter-villin distance of %s' % dbf)
        update_villin_closest(dbf, parm_n)

    # r2 closest villin to glm
    if 1:
        print('Update glm to closest 50 villin distance of %s' % dbf)
        update_glm_villin_closest(dbf, parm_n)

    parm_n = 5
    # r3 furthest glm-villin distance
    if 1:
        print('Update glm-villin distance of %s' % dbf)
        update_x_to_villin_avg_f5(dbf, mtype='glm', num=parm_n)

    # r4 furthest villin-villin distance
    if 1:
        print('Update inter-villin distance of %s' % dbf)
        update_x_to_villin_avg_f5(dbf, mtype='tubule', num=parm_n)

    cur = db.execute('pragma journal_mode=delete;')
    row = cur.fetchone()
    print('result of switch pragma journal_mode to delete = ', row[0])
    db.close()


if __name__ == "__main__":
    main()
