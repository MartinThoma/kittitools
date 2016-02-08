#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parse XML files containing tracklet info for kitti data base.

Parses the raw data section (http://cvlibs.net/datasets/kitti/raw_data.php).

example usage:

    import parseTrackletXML as xmlParser
    kitti_dir = '/path/to/kitti/data'
    drive = '2011_09_26_drive_0001'
    xmlParser.example(kitti_dir, drive)

or simply on command line:

    python parseTrackletXML.py
"""


from sys import argv
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools
from warnings import warn

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}

OCC_UNSET = 255    # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1': OCC_UNSET,
               '0': OCC_VISIBLE,
               '1': OCC_PARTLY,
               '2': OCC_FULLY}

TRUNC_UNSET = 255    # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99': TRUNC_UNSET,
                 '0': TRUNC_IN_IMAGE,
                 '1': TRUNC_TRUNCATED,
                 '2': TRUNC_OUT_IMAGE,
                 '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
    """Representation an annotated object track.

    Tracklets are created in function parse_xml and can most conveniently used
    as follows:

    for trackletObj in parse_xml(trackletFile):
        for translation, rotation, state, occlusion, truncation, amtOcclusion,
            amtBorders, absoluteFrameNumber in trackletObj:
            ... your code here ...

    absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
    amtOcclusion and amtBorders could be None

    You can of course also directly access the fields objType (string),
    size (len-3 ndarray), firstFrame/nFrames (int),
    trans/rots (nFrames x 3 float ndarrays),
    states/truncs (len-nFrames uint8 ndarrays),
    occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray)
    and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
    """

    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None  # n x 3 float array (x,y,z)
    rots = None  # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None  # n x 2 uint8 array    (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation

    # None or (n x 2) float array (amt_occlusion, amt_occlusion_kf)
    amtOccs = None

    amtBorders = None  # None (n x 3) float array    (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        """Create Tracklet with no info set."""
        self.size = np.nan*np.ones(3, dtype=float)

    def __str__(self):
        """Return human-readable string representation of tracklet object.

        called implicitly in
        print(trackletObj)
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.nFrames,
                                                           self.objectType)

    def __iter__(self):
        """
        Return iterator that yields tuple of all available data for each frame.

        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation, amtOcclusion,
            amtBorders, absoluteFrameNumber in trackletObj:
            ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amtOccs is None:
            return itertools.izip(self.trans, self.rots, self.states,
                                  self.occs, self.truncs,
                                  itertools.repeat(None),
                                  itertools.repeat(None),
                                  xrange(self.firstFrame,
                                         self.firstFrame+self.nFrames))
        else:
            return itertools.izip(self.trans, self.rots, self.states,
                                  self.occs, self.truncs,
                                  self.amtOccs, self.amtBorders,
                                  xrange(self.firstFrame,
                                         self.firstFrame+self.nFrames))


def parse_xml(_tracklet_file):
    """
    Parse tracklet xml file and convert results to list of Tracklet objects.

    Parameters
    ----------
    _tracklet_file : str
        name of a tracklet xml file

    Returns
    -------
    list of Tracklet objects read from xml file
    """
    # convert tracklet XML data to a tree structure
    etree = ElementTree()
    print('parsing tracklet file', _tracklet_file)
    with open(_tracklet_file) as f:
        etree.parse(f)

    # now convert output to list of Tracklet objects
    tracklets_elem = etree.find('tracklets')
    tracklets = []
    tracklet_idx = 0
    n_tracklets = None
    for trackletElem in tracklets_elem:
        # print('track:', trackletElem.tag)
        if trackletElem.tag == 'count':
            n_tracklets = int(trackletElem.text)
            print('file contains', n_tracklets, 'tracklets')
        elif trackletElem.tag == 'item_version':
            pass
        elif trackletElem.tag == 'item':
            # print('tracklet {0} of {1}'.format(tracklet_idx, n_tracklets))
            # a tracklet
            new_track = Tracklet()
            is_finished = False
            has_amt = False
            frame_idx = None
            for info in trackletElem:
                # print('trackInfo:', info.tag)
                if is_finished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    new_track.objectType = info.text
                elif info.tag == 'h':
                    new_track.size[0] = float(info.text)
                elif info.tag == 'w':
                    new_track.size[1] = float(info.text)
                elif info.tag == 'l':
                    new_track.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    new_track.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        # print('trackInfoPose:', pose.tag)

                        # this should come before the others
                        if pose.tag == 'count':
                            if new_track.nFrames is not None:
                                raise ValueError('there are several pose '
                                                 'lists for a single track!')
                            elif frame_idx is not None:
                                raise ValueError('?!')
                            new_track.nFrames = int(pose.text)
                            new_track.trans = np.nan * np.ones((new_track.nFrames, 3), dtype=float)
                            new_track.rots = np.nan * np.ones((new_track.nFrames, 3), dtype=float)
                            new_track.states = np.nan * np.ones(new_track.nFrames, dtype='uint8')
                            new_track.occs = np.nan * np.ones((new_track.nFrames, 2), dtype='uint8')
                            new_track.truncs = np.nan * np.ones(new_track.nFrames, dtype='uint8')
                            new_track.amtOccs = np.nan * np.ones((new_track.nFrames, 2), dtype=float)
                            new_track.amtBorders = np.nan * np.ones((new_track.nFrames, 3), dtype=float)
                            frame_idx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frame_idx is None:
                                raise ValueError('pose item came before number of poses!')
                            for poseInfo in pose:
                                # print('trackInfoPoseInfo:', poseInfo.tag)
                                if poseInfo.tag == 'tx':
                                    new_track.trans[frame_idx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    new_track.trans[frame_idx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    new_track.trans[frame_idx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    new_track.rots[frame_idx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    new_track.rots[frame_idx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    new_track.rots[frame_idx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    new_track.states[frame_idx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    new_track.occs[frame_idx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    new_track.occs[frame_idx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    new_track.truncs[frame_idx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    new_track.amtOccs[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    new_track.amtOccs[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    new_track.amtBorders[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    new_track.amtBorders[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    new_track.amtBorders[frame_idx, 2] = float(poseInfo.text)
                                    has_amt = True
                                else:
                                    raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frame_idx += 1
                        else:
                            raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    is_finished = True
                else:
                    raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))

            # some final consistency checks on new tracklet
            if not is_finished:
                warn('tracklet {0} was not finished!'.format(tracklet_idx))
            if new_track.nFrames is None:
                warn('tracklet {0} contains no information!'.format(tracklet_idx))
            elif frame_idx != new_track.nFrames:
                warn(('tracklet {0} is supposed to have {1} frames, '
                      'but perser found {1}!').format(tracklet_idx,
                                                      new_track.nFrames,
                                                      frame_idx))
            if np.abs(new_track.rots[:, :2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amtOccs / amtBorders are not set, set them to None
            if not has_amt:
                new_track.amtOccs = None
                new_track.amtBorders = None

            # add new tracklet to list
            tracklets.append(new_track)
            tracklet_idx += 1

        else:
            raise ValueError('unexpected tracklet info')

    print('loaded', tracklet_idx, 'tracklets')

    # final consistency check
    if tracklet_idx != n_tracklets:
        warn(('according to xml information the file has {0} tracklets, but '
              'parser found {1}!').format(n_tracklets, tracklet_idx))

    return tracklets


def example(kitti_dir=None, drive=None):
    """
    Example how to use this script (TODO?).

    Parameters
    ----------
    kitti_dir : None
    drive : None
    """
    from os.path import join, expanduser
    import readline  # makes raw_input behave more fancy
    # from xmlParser import parse_xml, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

    default_drive = '2011_09_26_drive_0001'

    # get dir names
    if kitti_dir is None:
        kitti_dir = expanduser(raw_input('please enter kitti base dir '
                                         '(e.g. ~/path/to/kitti): ').strip())
    if drive is None:
        drive = raw_input(('please enter drive name '
                           '(default {0}): ').format(default_drive)).strip()
        if len(drive) == 0:
            drive = default_drive

    # read tracklets from file
    my_tracklet_file = join(kitti_dir, drive, 'tracklet_labels.xml')
    tracklets = parse_xml(my_tracklet_file)

    # loop over tracklets
    for iTracklet, tracklet in enumerate(tracklets):
        print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        # this part is inspired by kitti object development kit matlab code:
        # computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation
        # yet\
        tracklet_box = np.array([[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                                 [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                                 [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # loop over all data in tracklet
        for (translation, rotation, state, occlusion, truncation, amtOcclusion,
             amtBorders, absoluteFrameNumber) in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            # other rotations are 0 in all xml files I checked
            yaw = rotation[2]
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                               [np.sin(yaw), np.cos(yaw), 0.0],
                               [0.0, 0.0, 1.0]])
            corner_pos_in_velo = (np.dot(rot_mat, tracklet_box) +
                                  np.tile(translation, (8, 1)).T)

            # calc yaw as seen from the camera (i.e. 0 degree = facing away
            # from cam), as opposed to
            #     car-centered yaw (i.e. 0 degree = same orientation as car).
            #     makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yaw_visual = (yaw - np.arctan2(y, x)) % (2.*np.pi)


# when somebody runs this file as a script:
#     run example if no arg or only 'example' was given as arg
#     otherwise run parse_xml
if __name__ == "__main__":
    # argv[0] is 'parseTrackletXML.py'
    if len(argv) < 2:
        example()
    elif (len(argv) == 2) and (argv[1] == 'example'):
        example()
    else:
        parse_xml(*argv[1:])
