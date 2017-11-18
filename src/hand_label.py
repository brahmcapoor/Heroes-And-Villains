import collections

DATASET_PATH = "../dataset/"

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

def parse_data_from_files():
    m_ids_to_titles     = parse_title_data(DATASET_PATH + 'movie_titles_metadata.txt')
    m_ids_to_characters = parse_character_data(DATASET_PATH + 'movie_characters_metadata.txt')
    m_ids_to_pa_pairs   = parse_existing_pairs(DATASET_PATH + 'movie_pa_labels.txt')
    return m_ids_to_titles, m_ids_to_characters, m_ids_to_pa_pairs

def remove_previously_labeled(m_ids_to_titles, m_ids_to_pa_pairs):
    for m_id, title in m_ids_to_pa_pairs.items():
        del m_ids_to_titles[m_id]

def print_prompt(m_title, c_list):
    print ("~~~~ " * 10)
    print ("MOVIE: {} \nCHARACTERS:".format(m_title))
    for c_index, c_info in enumerate(c_list):
        c_id, c_name = c_info
        print ("  {}) {} [id:{}]".format(c_index, c_name, c_id))

def get_character_choice(c_list, prompt):
    while True:
        try:
            index = input(prompt)
            index = int(index)
        except ValueError:
            print ("Invalid input (value error). Try again.")
            continue
        if index == -1:
            return None
        if index in range(len(c_list)):
            return c_list[index]
        print ("Invalid input (out of bounds). Try again.")

def verify_input(protagonist, antagonist, m_title):
    verify = input("VERIFY:\n  Protagonist: {}\n  Antagonist: {}\nPress ENTER if true; input any other character if false: ".format(
        protagonist[1] if protagonist else 'UNKNOWN',
        antagonist[1] if antagonist else 'UNKNOWN'))
    return not verify

def pa_pair_for_movie(m_id, m_title, m_ids_to_characters):
    c_list = list(m_ids_to_characters[m_id])
    print_prompt(m_title, c_list)
    while True:
        protagonist = get_character_choice(c_list, "Type the index of the protagonist (\'-1\' if unknown): ")
        antagonist  = get_character_choice(c_list, "Type the index of the antagonist (\'-1\' if unknown): ")
        if verify_input(protagonist, antagonist, m_title): return protagonist, antagonist

def write_pair_to_file(m_id, protagonist, antagonist):
    with open(DATASET_PATH + 'movie_pa_labels.txt', 'a') as f:
        if protagonist and antagonist:
            f.write(' +++$+++ '.join([m_id, protagonist[0], antagonist[0]]))
            f.flush()
            f.write('\n')
            f.flush()
            print ("Added to file!")
        else:
            print ("Skipped!")

def main():
    m_ids_to_titles, m_ids_to_characters, m_ids_to_pa_pairs = parse_data_from_files()
    remove_previously_labeled(m_ids_to_titles, m_ids_to_pa_pairs)
    for m_id, m_title in m_ids_to_titles.items():
        protagonist, antagonist = pa_pair_for_movie(m_id, m_title, m_ids_to_characters)
        write_pair_to_file(m_id, protagonist, antagonist)

if __name__ == '__main__':
    main()
