import os
import json
import subprocess
import shutil
import numpy as np
# pth = 'cali-test/data/cali-1/0000050'

# f_list = os.listdir(pth)
# s_list =list(set([f.split('-')[0] for f in f_list]))
# s_dict = {}
# for s in s_list:
#     s_dict[s[-4:]] = s
    
# output_fn = os.path.join(pth, "common.json")
# with open(output_fn, 'w') as f:
#     json.dump(s_dict, f, indent=4)
    
    
cam_series = {
    "1362": "049122251362",
    "1634": "043422251634",
    "4320": "151422254320",
    "1000": "215122251000",
    "1318": "213622251318",
    "1169": "035622251169",
    "2129": "152522252129",
    "1171": "213622301171",
    "0028": "035322250028",
    "8540": "234322308540",
    "1246": "043422251246",
    "1265": "035322251265",
    "1705": "105322251705",
    "1753": "234222301753",
    "1973": "235422301973",
    "0244": "035322250244",
    "1516": "138322251516",
    "1228": "035322251228",
    "1487": "043422251487",
    "1116": "035322251116",
    "0385": "038122250385",
    "2543": "043422252543",
    "0879": "046122250879",
    "0406": "035722250406",
    "2448": "117222252448",
    "1285": "035322251285",
    "1100": "046122251100",
    "1040": "213622301040",
    "0103": "234222300103"
}

camera_set = {
    0 : ["0385", "2543", "1246", "1973"],
    1 : ["4320", "1040", "1634"],
    2 : ["1705", "1318", "1100"],
    3 : ["1285", "1753", "8540"],
    4 : ["1116", "1265", "0103"],
    5 : ["1169", "1516", "2448"],
    6 : ["2129", "0028", "0244"],
    7 : ["1228", "0879", "1362"],
    8 : ["1171", "1000", "1487", "0406"],
}

d2c_offset = np.array([
    [ 9.999972581863403320e-01, -1.704507041722536087e-03, -1.608532853424549103e-03, -5.915403366088867188e-02 ],
    [ 1.697299536317586899e-03,  9.999885559082031250e-01, -4.471577703952789307e-03,  7.883417129050940275e-05 ],
    [ 1.616136287339031696e-03,  4.468835424631834030e-03,  9.999887347221374512e-01,  2.413551264908164740e-04 ],
    [ 0.000000000000000000e+00,  0.000000000000000000e+00,  0.000000000000000000e+00,  1.000000000000000000e+00 ]
])

keyframe_list = np.array([240, 220, 200, 175, 145, 110, 80, 50])
id2cam = ["1246","2543","0385","1973","1634","1040","4320","0879","1362","1228","0028","0244","2129","2448","1516","1169","0103","1265","1116","1753","8540","1285","1100","1318","1705"]
if __name__ == "__main__":
    # frames = os.listdir("data/cali_2/raw_data")
    # for f in frames:
    #     command = [
    #         'python', 'read_color_raw422_multi.py',
    #         '--input_dir', f'./data/cali_2/raw_data/{f}',
    #         '--output_dir', f'./data/cali_2/img_data/{f}',
    #     ]
    #     print(' '.join(command))
    #     subprocess.run(command, check=True)
    input_pth = "data/trans_data"
    output_pth = "data/transform"
    if not os.path.exists(output_pth):
        os.mkdir(output_pth)
    trans_list = os.listdir(input_pth)
    for trans in trans_list:
        trans_id = trans.split('_')
        if trans_id[2] == '1246':
            shutil.copy2(os.path.join(input_pth, trans), os.path.join(output_pth,trans))