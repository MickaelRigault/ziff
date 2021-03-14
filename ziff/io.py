
import os

from ztfquery.io import LOCALSOURCE

_PSFDIR = os.path.join(LOCALSOURCE,"psf")    
ZIFFDIR = os.path.join(_PSFDIR,"ziff")
if not os.path.isdir(ZIFFDIR):
    # if created by someone esle in the meantime
    os.makedirs(ZIFFDIR, exist_ok=True) 


def get_psf_suffix(config, baseline="psf", extension=".piff"):
    """ """
    modelname = config["psf"]["model"]["type"]
    interpname = config["psf"]["interp"]["type"]
    interporder = config["psf"]["interp"]["order"]
    return f'{baseline}_{modelname}_{interpname}{interporder}{extension}'

def parse_psf_suffix(pifffile, expand=False):
    """ """
    _, filefracday, paddedfield, filtercode, ccd_, imgtypecode, qid_, *suffix_ = os.path.basename(pifffile).split("_")
    return suffix_ if expand else "_".join(suffix_)


def get_digit_dir(subdir="", builddir=True):
    """ """
    dirout = os.path.join(ZIFFDIR,"digit")
        
    if subdir is not None and subdir != "":
        dirout = os.path.join(dirout, subdir)
    
    if builddir and not os.path.isdir(dirout):
        os.makedirs(dirout, exist_ok=True)
        
    return dirout


def get_filedataframe(files):
    """ """
    
    from ztfquery import buildurl, fields
    
    def read_filename(file):
        _, filefracday,paddedfield, filtercode, ccdid_, _, qid_, suffix =  file.split("/")[-1].split("_")
        year,month, day, fracday = buildurl.filefrac_to_year_monthday_fracday(filefracday)
        ccdid = int(ccdid_[1:])
        qid = int(qid_[1:])        
        return {"obsdate": f"{year}-{month}-{day}",
                "fracday":fracday,
                "filefracday":filefracday,
                "fieldid":int(paddedfield),
                "filtername":filtercode,
                "ccdid":ccdid,
                "qid":qid,
                "rcid":fields.ccdid_qid_to_rcid(ccdid,qid),
                "suffix":suffix,
                "filename":file
               }

    return pandas.DataFrame([read_filename(f) for f in files])
