import os, time
import urllib
from multiprocessing import Pool
from tqdm import tqdm

def load_image_list(url_path):
    """
        Reads the image list file in url_path
        and returns a dictionnary of the form {class:[list of urls]}
    """
    with open(url_path, "r", encoding="latin-1") as f:
        urls = f.readlines() 
        urls = [str(url.strip()) for url in urls]

    dico = {}
    for url in tqdm(urls):
        try:
            idx, url = url.split("\t")
        except Exception:
            pass
        wnid = idx.split("_")[0]
        if wnid in dico:
            dico[wnid].append((wnid, idx, url))
        else:
            dico[wnid]=[(wnid, idx, url)]
    return dico

def dwn(x):
    """
        
    """
    wnid, fname, url, path = x
    if url.strip().split(".")[-1].lower() in ['jpg', 'gif', 'jpeg','png', 'bmp', 'thb', 'jpe', 'tif', 'pjpeg']:
        fname = fname + "." + url.strip().split(".")[-1] # keep the file extension in fname 
    else: 
        fname = fname + ".jpg" # default use .jpg file extension
    start = time.time()
    try:
        request = urllib.request.urlopen(url, timeout=5)
        image = request.read()
        with open(os.path.join(path, fname), 'wb') as f:
            f.write(image)
        return (None, time.time() - start)
    except Exception as e:
        return (type(e), time.time() - start)

def download_test_images(output_folder, dico, nprocess=10):
    """
    
    """
    pool=Pool(nprocess)
    for wnid in tqdm(dico):
        path = os.path.join(output_folder, wnid)
        if not os.path.isdir(path):
            os.makedirs(path)
        inp = list(map(lambda x: x+(path,), dico[wnid]))
        x = pool.map(dwn, inp)        