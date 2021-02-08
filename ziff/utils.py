



""" Generic tools """
import numpy as np

def is_running_from_notebook():
    """ Test if currently ran in notebook """
    return running_from() == "notebook"

def running_from():
    """  Where is the code running from ?"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return "notebook"   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return "terminal"  # Terminal running IPython
        else:
            return "other"  # Other type (?)
    except NameError:
        return None 

    
def avoid_duplicate(string_array):
    """ """
    string_array = np.asarray(string_array, dtype=object)
    occurence = {k:v for k,v in zip(*np.unique(string_array, return_counts=True))}
    for name_,nname_ in occurence.items():
        if nname_>1:
            string_array[np.in1d(string_array,name_)] = [name_+f"{i+1}" for i in range(nname_)]
            
    return np.asarray(string_array, dtype="str")

def vminvmax_parser(data_, vmin, vmax):
    """ """
    if vmin is None:
        vmin="0"
    if vmax is None:
        vmax = "100"
    if type(vmin) == str:
        vmin = np.percentile(data_, float(vmin))
    if type(vmax) == str:
        vmax = np.percentile(data_, float(vmax))
        
    return vmin, vmax
