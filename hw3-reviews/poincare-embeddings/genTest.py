from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.viz.poincare import poincare_2d_visualization
import plotly.io as pio



from plotly.offline import init_notebook_mode, iplot

MODEL_NAME = "model500.test"
# relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
# model = PoincareModel(relations, negative= 2)
# model.train(epochs = 50)

# file_path = datapath('poincare_hypernyms.tsv')
file_path = datapath('/home/taco650/ECE/ECE194N/ece594n-fork/hw3-reviews/poincare-embeddings/wordnet/mammal_closure_noweights.csv')
model = PoincareModel(PoincareRelations(file_path, delimiter=','), alpha = 0.1, burn_in_alpha=0.01, size = 2, workers = 1, burn_in = 40, negative=10)
model.train(epochs=500,print_every=10)
model.save(MODEL_NAME)

model = PoincareModel.load(MODEL_NAME)
wv = model.kv.get_vector('kangaroo.n.01')
print(wv)
# wv.save('vectors.kv')
# reloaded_word_vectors = KeyedVectors.load('vectors.kv')
print(model.kv.similarity('mammal.n.01', 'carnivore.n.01'))

fig = poincare_2d_visualization(model, PoincareRelations(file_path, delimiter=','), "Test", num_nodes=50, show_node_labels=())
pio.show(fig)
print(type(fig))
# iplot(fig)
# py.offline.image.ishow(fig,width=1000,height=1000)