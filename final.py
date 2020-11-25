from matplotlib.pyplot import imread, show, figure, title, savefig
from numpy.linalg import svd
from numpy import zeros, max, min, round, ndarray, asarray, reshape
from math import log2
from struct import pack, unpack
from cv2 import imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, cvtColor, imwrite
from queue import PriorityQueue
import tkinter as tk
from tkinter import filedialog

def zip_image_by_svd(origin_image, rate=0.8):

    result = zeros(origin_image.shape)
    metrac = []
    u_shape = 0
    s_shape = 0
    vT_shape = 0

    for chan in range(3):
        U, sigma, V = svd(origin_image[:, :, chan])
        n_sigmas = 0
        temp = 0

        while (temp / sum(sigma)) < rate:
            temp += sigma[n_sigmas]
            n_sigmas += 1

        S = zeros((n_sigmas, n_sigmas))

        for i in range(n_sigmas):
            S[i, i] = sigma[i]

        metrac.append((list(U[:, 0:n_sigmas]), list(S), list(V[0:n_sigmas, :])))
        result[:, :, chan] = (U[:, 0:n_sigmas].dot(S)).dot(V[0:n_sigmas, :])
        u_shape = U[:, 0:n_sigmas].shape
        s_shape = S.shape
        vT_shape = V[0:n_sigmas, :].shape

    for i in range(3):
        MAX = max(result[:, :, i])
        MIN = min(result[:, :, i])
        result[:, :, i] = (result[:, :, i] - MIN) / (MAX - MIN)

    result = round(result * 255).astype('int')

    zip_rate = (origin_image.size - 3 * (u_shape[0] * u_shape[1] + s_shape[0]
                                         * s_shape[1] + vT_shape[0] * vT_shape[1])) / (origin_image.size)

    return [result, rate, n_sigmas, origin_image.shape,
            (u_shape, s_shape, vT_shape), zip_rate, metrac]


class HuffmanNode(object):


    def __init__(
            self,
            value,
            key=None,
            symbol='',
            left_child=None,
            right_child=None):

        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.key = key
        assert symbol == ''
        self.symbol = symbol

    def __eq__(self, other):

        return self.value == other.value

    def __gt__(self, other):

        return self.value > other.value

    def __lt__(self, other):

        return self.value < other.value


def createTree(hist_dict: dict) -> HuffmanNode:

    q = PriorityQueue()

    for k, v in hist_dict.items():
        q.put(HuffmanNode(value=v, key=k))

    while q.qsize() > 1:
        l_freq, r_freq = q.get(), q.get()
        node = HuffmanNode(
            value=l_freq.value +
            r_freq.value,
            left_child=l_freq,
            right_child=r_freq)
        q.put(node)

    return q.get()


def walkTree_VLR(root_node: HuffmanNode, symbol=''):

    global Huffman_encode_dict

    if isinstance(root_node, HuffmanNode):
        root_node.symbol += symbol
        if root_node.key is not None:
            Huffman_encode_dict[root_node.key] = root_node.symbol

        walkTree_VLR(root_node.left_child, symbol=root_node.symbol + '0')
        walkTree_VLR(root_node.right_child, symbol=root_node.symbol + '1')
    return


def encodeImage(src_img: ndarray, encode_dict: dict):

    img_encode = ""
    assert len(src_img.shape) == 1, '`src_img` must be a vector'
    for pixel in src_img:
        img_encode += encode_dict[pixel]
    return img_encode


def writeBinImage(img_encode: str, huffman_file: str):

    with open(huffman_file, 'wb') as f:
        for i in range(0, len(img_encode), 8):
            img_encode_dec = int(img_encode[i:i + 8], 2)
            img_encode_bin = pack('>B', img_encode_dec)
            f.write(img_encode_bin)


def readBinImage(huffman_file: str, img_encode_len: int):
    code_bin_str = ""
    with open(huffman_file, 'rb') as f:
        content = f.read()
        code_dec_tuple = unpack('>' + 'B' * len(content), content)
        for code_dec in code_dec_tuple:
            code_bin_str += bin(code_dec)[2:].zfill(8)
        len_diff = len(code_bin_str) - img_encode_len
        code_bin_str = code_bin_str[:-8] + code_bin_str[-(8 - len_diff):]
    return code_bin_str


def decodeHuffman(img_encode: str, huffman_tree_root: HuffmanNode):
    img_src_val_list = []

    root_node = huffman_tree_root
    for code in img_encode:
        if code == '0':
            root_node = root_node.left_child
        elif code == '1':
            root_node = root_node.right_child
        if root_node.key is not None:
            img_src_val_list.append(root_node.key)
            root_node = huffman_tree_root
    return asarray(img_src_val_list)


def decodeHuffmanByDict(img_encode: str, encode_dict: dict):
    img_src_val_list = []
    decode_dict = {}
    for k, v in encode_dict.items():
        decode_dict[v] = k
    s = 0
    while len(img_encode) > s + 1:
        for k in decode_dict.keys():
            if k == img_encode[s:s + len(k)]:
                img_src_val_list.append(decode_dict[k])
                s += len(k)
                break
    return asarray(img_src_val_list)

def walkTree_VLR(root_node: HuffmanNode, symbol=''):
    global Huffman_encode_dict

    if isinstance(root_node, HuffmanNode):
        root_node.symbol += symbol
        if root_node.key is not None:
            Huffman_encode_dict[root_node.key] = root_node.symbol

        walkTree_VLR(root_node.left_child, symbol=root_node.symbol + '0')
        walkTree_VLR(root_node.right_child, symbol=root_node.symbol + '1')
    return


def file_select():
    global now_file_name
    global origin_image
    global origin_code
    if str(entry_name.get()) == '':
        filename = tk.filedialog.askopenfilename()
    else:
        filename = entry_name.get()
    now_file_name = filename
    dataType = str(filename).split('.')[-1]

    if dataType == 'jpg':
        try:
            origin_image = imread(str(filename))
            lb.config(text="读取成功，您选择的文件是：" + filename)
            entry_name__var.set(str(filename))
            imshow("Origin Image", origin_image)
        except Exception:
            lb.config(text="图片读取失败，请检查图片文件地址或者文件名")
            entry_name__var.set(str(filename))
    elif dataType == 'txt':
        try:
            with open(str(filename)) as f:
                origin_code = f.read()
            lb.config(text="读取成功，您选择的文件是：" + filename)
            entry_name__var.set(str(filename))
        except Exception:
            lb.config(text="编码读取失败，请检查编码文件地址或者文件名")
            entry_name__var.set(str(filename))
        print(origin_code)
    else:
        lb.config(text="导入失败 ！请输入 .txt 或 .jpg 文件")
        entry_name__var.set(str(filename))


def svd_encode_decode():
    pic_file_name = "None"
    code_file_name = "None"
    if len(origin_image) == 0:
        out_mess.config(text="未导入文件 ！请输入 .txt 或 .jpg 文件")
        out = ["None", "None", "None", "None",
               ["None", "None", "None"], "None"]
    else:
        try:
            if str(entry_svd.get()) == '':
                out = zip_image_by_svd(origin_image)
            else:
                out = zip_image_by_svd(
                    origin_image, rate=float(str(entry_svd.get())))
            first_name = str(now_file_name).split('.')[0]
            pic_file_name = first_name + "SVD_Decoded_pic.jpg"
            code_file_name = first_name + "SVD_Encoded_code.txt"
            imshow("Decoded Image", out[0])
            imwrite(pic_file_name, out[0])
            # show()
            with open(code_file_name, 'w') as f:
                f.write(str(list(out[-1])))
                f.close()
        except Exception:
            print("hh")
            out_mess.config(text="图像压缩出错，请更换图片 ！")
            out = [
                "None", "None", "None", "None", [
                    "None", "None", "None"], "None"]

    out_mess.config(text="\
奇异值保留率：{}\n\
所用奇异值数量为：{}\n\
原图大小：{}\n\
压缩后用到的矩阵大小：3 X {}\n\
压缩率为：{}\n\n\
SVD编码后的编码保存为{}文件\n\
SVD解码后的图像保存为{}文件".format(out[1], out[2], out[3], out[4], out[5],
                         code_file_name, pic_file_name))


def huff_encode_decode():
    src_img = cvtColor(origin_image, COLOR_BGR2GRAY)
    src_img_w, src_img_h = src_img.shape
    src_img_ravel = src_img.ravel()

    hist_dict = {}
    global Huffman_encode_dict

    for p in src_img_ravel:
        if p not in hist_dict:
            hist_dict[p] = 1
        else:
            hist_dict[p] += 1

    huffman_root_node = createTree(hist_dict)
    walkTree_VLR(huffman_root_node)

    img_encode = encodeImage(src_img_ravel, Huffman_encode_dict)
    writeBinImage(
        img_encode,
        now_file_name.replace(
            '.jpg',
            'Huffman_Encoded_code.bin'))

    img_read_code = img_encode
    img_src_val_array = decodeHuffman(img_read_code, huffman_root_node)
    assert len(img_src_val_array) == src_img_w * src_img_h
    img_decode = reshape(img_src_val_array, [src_img_w, src_img_h])
    imshow("src_img", src_img)
    imshow("img_decode", img_decode)
    waitKey(0)
    destroyAllWindows()
    imwrite(now_file_name.replace('.jpg', 'Huffman_Dncoded_pic.jpg'),
            img_decode)
    total_code_len = 0
    total_code_num = sum(hist_dict.values())
    avg_code_len = 0
    I_entropy = 0
    for key in hist_dict.keys():
        count = hist_dict[key]
        code_len = len(Huffman_encode_dict[key])
        prob = count / total_code_num
        avg_code_len += prob * code_len
        I_entropy += -(prob * log2(prob))
    S_eff = I_entropy / avg_code_len

    first_name = str(now_file_name).split('.')[0]
    pic_file_name = first_name + "Huffman_Dncoded_pic.jpg"
    code_file_name = first_name + "Huffman_Encoded_code.txt"
    out_mess.config(text="平均编码长度为：{:.3f}\n编码效率为：{:.6f}\n\
SVD编码后的编码保存为了{}文件\n\
SVD解码后的图像保存为了{}文件".format(avg_code_len, S_eff, code_file_name, pic_file_name))


win_size = "800x600"
origin_image = []
origin_code = []
now_file_name = "h"
Huffman_encode_dict = {}


window = tk.Tk()
window.title('图像压缩编码/解码')
window.geometry(win_size)

ex = "软件使用说明部分\n\n\
1. 数据输入方式说明。有两种方式，如果在地址栏输入文件地址，然后点击“导入数据”则按照地址直接读取文件;\n\
如果地址栏未输入任何字符，点击“导入数据”可从本地选择数据.\n注 : 只支持 .jpg 文件输入\n\
2. 本软件可对 jpg 图片进行 SVD 压缩编解码和哈夫曼编解码码两种操作\n\n\
作者：方正豪  2020234248  1102047640@qq.com"
ex_text = tk.StringVar()
ex_label = tk.Label(
    window,
    textvariable=ex_text,
    bg='lightGray',
    width=110,
    height=8, justify="left")
ex_text.set(ex)  # \n输入文件地址/从电脑选择文件
ex_label.place(x=10, y=20, anchor='nw')

y_loc_input = 180
x_loc_input = 50
input_text = tk.StringVar()
input_label = tk.Label(
    window,
    textvariable=input_text,
    bg='lightGray',
    width=20,
    height=1)
input_text.set("文件输入（图像/编码）：")  # \n输入文件地址/从电脑选择文件
input_label.place(x=x_loc_input, y=y_loc_input, anchor='nw')

entry_name__var = tk.StringVar()
entry_name = tk.Entry(window, textvariable=entry_name__var, width=50)
entry_name.place(x=x_loc_input + 150, y=y_loc_input + 1, anchor='nw')

button_sample_data = tk.Button(window, text='导入数据', command=file_select)
button_sample_data.place(x=x_loc_input + 520, y=y_loc_input - 3, anchor='nw')

lb = tk.Label(window, text='')
lb.place(x=x_loc_input, y=y_loc_input + 30)


y_loc_code = 280
x_loc_code = 50
svd_text = tk.StringVar()
svd_label = tk.Label(
    window,
    textvariable=svd_text,
    bg='lightGray',
    width=12,
    height=1)
svd_text.set("SVD编码/解码 :")
svd_label.place(x=x_loc_code, y=y_loc_code, anchor='nw')

entry_svd__var = tk.StringVar()
entry_svd = tk.Entry(window, textvariable=entry_svd__var, width=10)
entry_svd.place(x=x_loc_code + 95, y=y_loc_code + 1, anchor='nw')

button_svd = tk.Button(window, text='SVD运行', command=svd_encode_decode)
button_svd.place(x=x_loc_code + 180, y=y_loc_code - 3, anchor='nw')

lba = tk.Label(window, text='')
lba.place(x=x_loc_code, y=y_loc_code + 30)
y_loc_huff = 280
x_loc_huff = 430

huff_text = tk.StringVar()
huff_label = tk.Label(
    window,
    textvariable=huff_text,
    bg='lightGray',
    width=16,
    height=1)
huff_text.set("Huffman编码/解码 :")
huff_label.place(x=x_loc_huff, y=y_loc_huff, anchor='nw')

button_huff = tk.Button(window, text='Huffman运行', command=huff_encode_decode)
button_huff.place(x=x_loc_huff + 125, y=y_loc_huff - 3, anchor='nw')



y_loc_out = 350
x_loc_out = 50

out_text = tk.StringVar()
out_label = tk.Label(
    window,
    textvariable=out_text,
    bg='lightGray',
    width=60,
    height=1)
out_text.set("编码/解码结果输出部分 :")
out_label.place(x=x_loc_out, y=y_loc_out, anchor='nw')

out_mess = tk.Label(window, text='hh', justify="left")
out_mess.place(x=x_loc_out, y=y_loc_out + 30)

window.mainloop()