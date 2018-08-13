import gensim


def log(word):
    print('{0} with prob: {1:2f}'.format(word[0], word[1]))


# Loading word2Vec model
w2v_model = gensim.models.Word2Vec.load('./model_18394.model')

# try top 5 similar_by_word
tuples = w2v_model.wv.similar_by_word('điện_thoại', topn=5)

"""
output
điện_thoại_di_động: 0.786089
máy: 0.677943
radio: 0.639867
di_động: 0.629827
máy_ảnh: 0.607558
"""
for word in tuples:
    log(word)

# does not match
# output: con_gái
print("====================================================================")
print("doesnt_match of: [điện_thoại iphone ipad con_gái]")
print(w2v_model.wv.doesnt_match("điện_thoại iphone ipad con_gái".split()))

# output: sách
print("doesnt_match of: [sách lập_trình máy_tính phần_mềm]")
print(w2v_model.wv.doesnt_match("sách lập_trình máy_tính phần_mềm".split()))

# output: mềm
print("doesnt_match of: [tím xanh đỏ mềm]")
print(w2v_model.wv.doesnt_match("tím xanh đỏ mềm".split()))

print("====================================================================")
