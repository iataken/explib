import pickle
def get_conf_cycle(dict_file):
    dic_file = open(dict_file,'rb')
    dic = pickle.load(dic_file)
    dic_file.close()

    conf_cycle = {}
    for author, feature in dic.iteritems():
        #start_year = 2020
        #end_year   = 1800
        if author == '':
            continue
        for each in feature:
            conf = each['conf']
            year = int(each['year'][0])

            if 'PKDD' == conf or 'ECML/PKDD (1)' == conf or 'ECML/PKDD (2)' == conf or 'ECML/PKDD (3)'== conf:
                conf = 'PKDD'
            elif 'PAKDD' ==  conf or 'PAKDD (2)'== conf or 'PAKDD (1)'== conf:
                conf = 'PAKDD'

            if conf not in conf_cycle:
                conf_cycle[conf]={}
                conf_cycle[conf]['start'] = 2020
                conf_cycle[conf]['end'] = 1800

            if conf_cycle[conf]['start'] > year:
                conf_cycle[conf]['start'] = year
            if conf_cycle[conf]['end'] < year:
                conf_cycle[conf]['end'] = year

    return conf_cycle








