import sys, os
import shutil
from pprint import pprint
import json
import pickle
from pathlib import Path

#sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#from misc import output_error_msg
import numpy as np

from settings import Settings

# class CustomEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, np.int32): 
#             return int(obj)  
#         return json.JSONEncoder.default(self, obj)
    
    
class CustomEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)




# -- dump json output --
def json_dump(fname, data, indent=4):
    global_settings = Settings()
    if global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
        print(f'Testing search time, skipping saving json data to {fname}')
        return

    # delete file if exists
    if os.path.exists(fname):
        os.remove(fname)
    
    # write json
    with open(fname, "w") as write_file:
        json.dump(data, write_file, indent=indent, cls=CustomEncoder)


# -- load data from json file --
def json_load(fname):
    
    if os.path.exists(fname):
        if fname.endswith('.zst'):
            import zstandard
            json_data=zstandard.open(fname)
        else:
            json_data=open(fname)
        file_data = json.load(json_data)
        return file_data
    else:
        sys.exit("ERROR - file does not exist : " + fname)
        return None


def open_file(fname):
    #print ("Opening - "+ fname)
    f = open(fname, 'r', encoding='utf-8') 
    data = f.readlines()
    f.close()
    return data

# def write_json_file(fname, data):                
#         logfile=open(fname, 'w')
#         json_data = json.dumps(data, indent=4)
#         logfile.write(json_data)


def write_file(fname, data):
    global_settings = Settings()
    if global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
        print(f'Testing search time, skipping saving data to {fname}')
        return

    f = open(fname, "wt")
    f.write(data)
    f.close()


def delete_file(fname):
    global_settings = Settings()
    if global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
        print(f'Testing search time, skipping removing file {fname}')
        return

    if os.path.exists(fname):
        os.remove(fname)
    else:
        pass

def copy_file(src, dst):
    if os.path.exists(src):
        try:
            shutil.copyfile(src, dst)
        except:
            #output_error_msg("File did not copy ! - \n" + dst + "\n" + src)
            raise BaseException("File did not copy ! - \n" + dst + "\n" + src)
            sys.exit()
    else:        
        raise BaseException('Src file invalid ! - ' + src)
        sys.exit()


def dir_create(dir):
    try:
        Path(dir).mkdir(parents=True, exist_ok=True)
    except:
        sys.exit("dir_create:: Error - " + dir)


def file_exists(fname):
    return os.path.exists(fname)


# def get_log_file_list(fname):
#     lines = open_file(fname)
#     result = []
#     for l in lines:
#         result.append(LOG_DIR + l)
#     return result

# return with full path
def get_dir_filelist(dirname):
    file_names = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    return file_names


# -- pickle handling
def pickle_dump(fname, data):
    global_settings = Settings()
    if global_settings.NAS_SETTINGS_GENERAL['SEARCH_TIME_TESTING']:
        print(f'Testing search time, skipping saving pickle data to {fname}')
        return

    with open(fname, 'wb') as file: 
        pickle.dump(data, file)    # serialize
        
def pickle_load(fname):  
    if file_exists(fname) == True:  
        with open(fname, 'rb') as file:         
            data = pickle.load(file) # deserialze  
        return data
    else:
        sys.exit("pickle_load:: Error - file doesn't exist: ", fname)
    
