from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('Plot the histogram of Flight column')
print(query_embedding)
passage_embedding = model.encode(['Here is a histogram of the flight column',
                                  'London is known for its finacial district',
                                  'The flight was late',
                                  'The flight was full',
                                  'The flight was empty',
                                  'The flight was cancelled',
                                  'The flight was delayed',
                                  'The flight was on time',
                                  'The flight was early',])
print(passage_embedding)

print("Similarity:", util.dot_score(query_embedding, passage_embedding))