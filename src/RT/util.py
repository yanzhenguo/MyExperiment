import pickle

temp_dir='../../temp/RT/util/'


def get_data():
    with open('/home/yan/my_datasets/RottonTomatoes/rt-polaritydata/rt-polarity.pos',encoding='latin') as f:
        text=f.readlines()
    with open('/home/yan/my_datasets/RottonTomatoes/rt-polaritydata/rt-polarity.neg',encoding='latin') as f:
        text+=f.readlines()
    with open(temp_dir+'text.pkl',mode='wb') as f:
        pickle.dump(text,f,1)
    print('read ',len(text),' lines')

if __name__=='__main__':
    get_data()
