def import_filepaths(image_ext,label_ext,meta_ext, folder='.\\data\\'):
    import glob
    image_filepaths = glob.glob(folder+'rgb\\*.'+image_ext) # Obtain filepaths of images
    label_filepaths = glob.glob(folder+'gt\\*.'+label_ext) 
    color_filepaths = glob.glob(folder+'gt_color\\*.' + label_ext)  
    meta_filepaths = glob.glob(folder+'meta\\*.' + meta_ext)  
    if image_filepaths == []:
        raise NameError("NO IMAGES FOUND")
    elif label_filepaths == []:
        raise NameError("NO LABELS FOUND")
    elif color_filepaths == []:
        raise NameError("NO COLOR LABELS FOUND")
    elif meta_filepaths == []:
        raise NameError("NO META DATA FOUND")
    return image_filepaths,label_filepaths, color_filepaths, meta_filepaths
