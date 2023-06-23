from itertools import groupby
def splitWithIndices(s, c=' '):
    p = 0
    for k, g in groupby(s, lambda x:x.isspace()):
        q = p + sum(1 for i in g)
        if not k:
         yield p, q # or p, q-1 if you are really sure you want that
        p = q


def normalization_code(code_src):
    words_index = list(splitWithIndices(code_src))   
    clean_word_index = []
    for w in words_index:
        if code_src[w[0]:w[1]].strip():
            clean_word_index.append(w)
    words_index = clean_word_index
    index_map = {}
    count = 0
    for c, w in enumerate(words_index):
        for i in range(w[0], w[1]):
            index_map[i]=count
            count += 1
        count+=1 # space between two words
     
    new_code = " ".join([ code_src[c:v] for (c,v) in words_index])
    return index_map, new_code

def alignment_tow_code(code1, code2):
    assert ''.join(code1.split()) == ''.join(code2.split()), f"\n {''.join(code1.split()) } \n {''.join(code2.split())}"
    code1_map = {}
    code2_map = {}
    count = 0
    for i, c1 in enumerate( code1 ):
        if c1.strip():
            code1_map[count] = i # non_space_character_pos, absolute_pos 
            count += 1
    assert count==len( ''.join(code1.split()) )
    count=0
    for j, c2 in enumerate( code2 ):
        if c2.strip():
            code2_map[count] = j
            count += 1
    assert count==len( ''.join(code2.split()) )
    code1_code2_map = {}
    for k in code1_map:
        code1_code2_map[code1_map[k]] = code2_map[k] 
    return code1_code2_map # absolute pos code1, absolute pos code2

def look_up( index_start, index_end, index_map):
    start_norm, end_norm = None, None
    look_start = True
    si = index_start
    while look_start:
        if si in index_map:
            start_norm = index_map[si]
            look_start = False
        else:
            si += 1

    look_end = True
    sj = index_end
    while look_end:
        if sj in index_map:
            end_norm = index_map[sj]
            look_end = False
        else:
            sj -= 1
    return start_norm, end_norm

def alignment(code_src, index_start, index_end):
    index_map, new_code = normalization_code(code_src)
    start_norm, end_norm = look_up(  index_start, index_end, index_map )
    return start_norm, end_norm, new_code