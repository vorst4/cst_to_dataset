import cv2
import dxfgrabber
import numpy as np


class _DrawingInterchangeFormat:

    def __init__(self):
        # settings , valid color range is from 0 to 255
        self.background_color = 0
        self.fill_color = 255
        self.dxf = None
        self.src = None

    def read(self, src):
        # read dxf file
        self.src = src
        self.dxf = dxfgrabber.readfile(src)

    def print(self):
        dxf = self.dxf
        print('  -------------------------------------------------------------------------------------------')
        print('  Info on: "' + str(self.src) + '"')
        print('  headers: ', dxf.header)
        print('  layers: ', len(dxf.layers))
        print('  blocks: ', len(dxf.blocks))
        print('  entities: ', len(dxf.entities))
        for ent in dxf.entities:
            print('    ' + ent.dxftype)
            ii = 0
            for point in ent.points:
                print('      [%i] % 10.5f % 10.5f % 10.5f' % (ii, point[0], point[1], point[2]))
                ii = ii + 1
        print('  -------------------------------------------------------------------------------------------')

    def export_as_png(self, dst, width, height, mm_per_px, plane):

        dxf = self.dxf

        # create empty image
        img = np.ones((height, width), np.uint8) * self.background_color

        # entities (rectangles, circles, polygons, etc.)
        for ent in dxf.entities:

            # extract points
            points = np.array(ent.points)

            # discard unused dimension
            if plane == 'xy':
                points = points[:, [0, 1]]
                # todo: check if the axes need to be swapped like in the 'xz' plane
            elif plane == 'xz':
                points = points[:, [0, 1]]
                # swap columns, so that x: width, z:height
                points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
            elif plane == 'yz':
                points = points[:, [0, 1]]
                # todo: check if the axes need to be swapped like in the 'yz' plane
            else:
                raise Exception('ERROR: value of <plane> is invalid: ' + plane)

            # points are currently in [mm], convert it to [px]
            points = points / mm_per_px

            # align coordinate origin (0, 0) with center of image
            points = points + np.array([height / 2, width / 2])

            # round [px] to int values
            points = points.astype(np.int32)

            # reshape to format that cv2 wants
            points = points.reshape((-1, 1, 2))

            # draw surface, depending on the type
            if ent.dxftype == 'POLYLINE':
                cv2.fillPoly(img, [points], color=self.fill_color)
            else:
                raise Exception('ERROR: encountered unknown entity ' + ent.dxftype)

        # save image
        cv2.imwrite(str(dst), img)
