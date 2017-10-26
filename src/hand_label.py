import collections

def parse_title_data(filename):
    m_ids_to_titles = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            tokens = line.split(' +++$+++ ')
            m_ids_to_titles[tokens[0]] = tokens[1]
    return m_ids_to_titles

def parse_character_data(filename):
    m_ids_to_characters = collections.defaultdict(set)
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            tokens = line.split(' +++$+++ ')
            m_ids_to_characters[tokens[2]].add((tokens[0], tokens[1]))
    return m_ids_to_characters

def parse_existing_pairs(filename):
    m_ids_to_pa_pairs   = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            tokens = line.split(' +++$+++ ')
            m_ids_to_pa_pairs[tokens[0]] = (tokens[1], tokens[2])
    return m_ids_to_pa_pairs

def main():
    m_ids_to_titles     = parse_title_data('movie_titles_metadata.txt')
    m_ids_to_characters = parse_character_data('movie_characters_metadata.txt')
    m_ids_to_pa_pairs   = parse_existing_pairs('movie_pa_labels.txt')

    with open('movie_pa_labels.txt', 'a') as f:
        for m_inf in m_ids_to_titles.items():
            m_id, m_title = m_inf
            if m_id in m_ids_to_pa_pairs: 
                continue

            for m_info in current_ms:
                m_id, m_title = m_info
                while True:
                    print ("[{}/{}]For the movie \'{}\' ({}), with the following characters:".format(len(m_ids_to_pa_pairs), len(m_ids_to_titles), m_title, m_id))
                    c_list = list(m_ids_to_characters[m_id])
                    for c_index, c_info in enumerate(c_list):
                        c_id, c_name = c_info
                        print ("  {}) {} ({})".format(c_index, c_name, c_id))
                    protagonist = input("Type the index of the protagonist (\'?\' if unkown): ")
                    if protagonist != '?':
                        protagonist = int(protagonist)
                    antagonist  = input("Type the index of the antagonist (\'?\' if unkown): ")
                    if antagonist != '?':
                        antagonist = int(antagonist)
                    verify = input("VERIFY: protagonist is {} and antagonist is {} (\'N\' if false): ".format(
                        c_list[protagonist][1] if protagonist != '?' else protagonist, 
                        c_list[antagonist][1]  if antagonist != '?' else antagonist))
                    if verify != 'N':
                        
                        if protagonist != "?" or antagonist != "?":
                            m_ids_to_pa_pairs[m_id] = (c_list[protagonist][0] if protagonist != '?' else -1, 
                                                   c_list[antagonist][0] if antagonist != '?' else -1)
                            f.write('{} +++$+++ {} +++$+++ {}\n'.format(
                                m_id, 
                                c_list[protagonist][0] if protagonist != '?' else -1, 
                                c_list[antagonist][0] if antagonist != '?' else -1)
                            )
                            f.flush()
                            print ("Added to file!")
                        else:
                            print ("Skipped!")
                        print ("") 
                        break

if __name__ == '__main__':
    main()