import os
import multiprocessing as mp
import glob
import argparse
# Manifold_path="/data3/haolin/Manifold/build/manifold"
def make_watertight(input_path,watertight_path,Manifold_path):
    os.system("timeout 10 %s %s %s 10000"%(Manifold_path,input_path,watertight_path))
    return

def run(future_root,save_root,Manifold_path):
    pool=mp.Pool(10)
    folder_list = os.listdir(future_root)
    for folder in folder_list:
        model_path_list=glob.glob(os.path.join(future_root,folder)+"/raw*.obj")
        if len(model_path_list)==0:
            continue
        else:
            model_path=model_path_list[0]

        #model_path=os.path.join(data_dir,folder,"raw_model.obj")
        if (".json" in model_path) or (".py" in model_path) or (".zip" in model_path):
            continue
        save_folder=os.path.join(save_root,folder)
        if os.path.exists(save_folder)==False:
            os.makedirs(save_folder)
        watertight_path=os.path.join(save_folder,"raw_watertight.obj")
        make_watertight(model_path,watertight_path,Manifold_path)
        #pool.apply_async(make_watertight,args=(model_path,watertight_path,Manifold_path))
        #make_watertight(model_path,save_path)
    pool.close()
    pool.join()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('generate watertight 3D-FUTURE model')
    parser.add_argument('--data_root', type=str,
                        help='root path of 3D-FUTURE model')
    parser.add_argument('--save_root', type=str, default='train', help='root path where to save the watertight model')
    parser.add_argument('--manifold_path', type=str, default='train', help='path where the manifold is installed')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    data_root=args.data_root
    save_root=args.save_root
    manifold_path=args.manifold_path
    run(data_root,save_root,manifold_path)


