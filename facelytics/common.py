def img_name_generator(ident_idx, img_idx):
    """
    Generate name for images in 00000_00.jpg
    """
    dst_name = f"{ident_idx:05d}_{img_idx:02d}.jpg"
    return dst_name
