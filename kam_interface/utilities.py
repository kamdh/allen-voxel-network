# Kameron Decker Harris
# modified from code by Nicholas Cain

def pickle(data, file_name):
    import pickle as pkl    
    f=open(file_name, "wb")
    pkl.dump(data, f)
    f.close()

def unpickle(file_name):
    import pickle as pkl
    f=open(file_name, "rb")
    data=pkl.load(f)
    f.close()
    return data

def write_dictionary_to_group(group, dictionary, create_name = None):
    if create_name != None:
        group = group.create_group(create_name)
    for key, val in dictionary.items():
        group[str(key)] = val
    return

def read_dictionary_from_group(group):
    dictionary = {}
    for name in group:
        dictionary[str(name)] = group[name].value
    return dictionary

def acro_list_to_id_list(ontology, acronyms):
    return [ontology.acronym_id_dict[name] for name in acronyms]

def absjoin(path,*paths):
    import os
    return os.path.abspath(os.path.join(path,*paths))
