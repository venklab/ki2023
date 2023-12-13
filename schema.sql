
CREATE TABLE IF NOT EXISTS Mask(
    id INTEGER PRIMARY KEY,
    pred_score REAL,  /* prediction score of mask */
    x INTEGER,
    y INTEGER,        /* upper left corner x,y map to Aperio Image */
    w INTEGER,
    h INTEGER,        /* width and height of mask */
    cx INTEGER,
    cy INTEGER,
    is_bad INTEGER DEFAULT 0,           /* bad mask */
    mask BLOB,                /* mask saved as RLE+zlib compress */
    mask_nuclei BLOB,         /* nuclei in glm */
    mask_area INTEGER,        /* non-zero pixel count on mask */
    mm_area INTEGER,          /* non-zero pixel count on segmentation */
    mm_mask BLOB,             /* mask for mesangial matrix */
    mm_pas BLOB,              /* PAS plus extracted nuclei */
    polygon BLOB,                /* mask saved as RLE+zlib compress */
    mm_opened BLOB,           /* mesangial matrix after open */
    mm_seg BLOB,              /* labels of mesangial matrix after open */

    note TEXT,                /* json format note */
    mme_score REAL    /* mesangial matrix expansion score */
);
CREATE INDEX IF NOT EXISTS idx_Mask_cx on Mask (cx);
CREATE INDEX IF NOT EXISTS idx_Mask_cy on Mask (cy);
CREATE INDEX IF NOT EXISTS idx_Mask_x on Mask (x);
CREATE INDEX IF NOT EXISTS idx_Mask_y on Mask (y);


/* some mask are very close to border, store them in this table temperoraly,
then extract a patch from level 0 slide image, centered at (cx, cy), run the
predict one more time for that region */
CREATE TABLE IF NOT EXISTS Secondpass (
    id INTEGER PRIMARY KEY,
    pred_score REAL,  /* prediction score of mask */
    cid INTEGER,      /* col slide window belongs */
    rid INTEGER,
    cx INTEGER,
    cy INTEGER        /* center x,y map to Aperio Image */

);


/* use as mask tiles cache for deepzoom */
CREATE TABLE IF NOT EXISTS Tile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    btile BLOB,      /* binary tile */
    tile BLOB        /* tile image saved as png */
);
CREATE INDEX IF NOT EXISTS idx_Tile_lid on Tile (lid);
CREATE INDEX IF NOT EXISTS idx_Tile_cid on Tile (cid);
CREATE INDEX IF NOT EXISTS idx_Tile_rid on Tile (rid);


/* slide tiles cache for labelwsi */
CREATE TABLE IF NOT EXISTS SlideTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile  BLOB        /* tile image saved as jpg blob, want small size */
);
CREATE INDEX IF NOT EXISTS idx_SlideTile_lid on SlideTile (lid);
CREATE INDEX IF NOT EXISTS idx_SlideTile_cid on SlideTile (cid);
CREATE INDEX IF NOT EXISTS idx_SlideTile_rid on SlideTile (rid);


/* mask tiles cache for labelwsi */
CREATE TABLE IF NOT EXISTS MaskTile(
    id INTEGER PRIMARY KEY,
    lid INTEGER,
    rid   INTEGER,
    cid   INTEGER,
    tile  BLOB,       /* tile image saved as png blob, need alpha */
    fullsize_mask BLOB
);
CREATE INDEX IF NOT EXISTS idx_MaskTile_lid on MaskTile (lid);
CREATE INDEX IF NOT EXISTS idx_MaskTile_cid on MaskTile (cid);
CREATE INDEX IF NOT EXISTS idx_MaskTile_rid on MaskTile (rid);


CREATE TABLE IF NOT EXISTS Info(
    id INTEGER PRIMARY KEY,
    title  TEXT,
    body   TEXT,
    maskfmt INTEGER   /* blob format used for store mask image, can be */
                      /* png or rle+compression */
);
