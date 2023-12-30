import os
import dominate
from dominate.tags import *

# create a html file to display the gif files of frame pairs

class HTML:
    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv='refresh', content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style='table-layout: fixed;')
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=256):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style='word-wrap: break-word;', halign='center', valign='top'):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style='width:%dpx' % (width), src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

def save_gif(image_paths, save_root):
    import imageio.v2 as imageio
    save_fn = '{}_{}.gif'.format(image_paths[0].split('/')[-2], os.path.basename(image_paths[0]).split('.')[0])
    
    with imageio.get_writer(os.path.join(save_root, save_fn), mode='I') as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    return save_fn
