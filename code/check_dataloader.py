from data_loader import MusicArrayLoader
import json

with open('model_config.json') as f:
    args = json.load(f)
dl = MusicArrayLoader(args['data_path'], args['time_step'], 64)

print(dl)
batch = dl.get_batch(args['batch_size'])
print(batch)

"""
# check dataloader output
for batch in MusicArrayLoader:
    print("roop")
    print(batch)
    
    ## check array ##
    # batch_ar = batch.numpy()
    # np.set_printoptions(threshold=np.inf)
    # print(batch_ar)
    
    break
"""