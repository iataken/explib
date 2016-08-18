import pickle

def get_author_cycle(dict_file):
    dic_file = open(dict_file,'rb')
    dic = pickle.load(dic_file)
    dic_file.close()

    author_cycle = {}
    for author, feature in dic.iteritems():
        if author == '':
            continue
        start_year = 2020
        end_year   = 1800
        for each in feature:
            year = int(each['year'][0])
            if start_year > year:
                start_year = year
            if end_year < year:
                end_year = year

        author_cycle[author] = {'start':start_year,
                                'end':end_year}

    return author_cycle








