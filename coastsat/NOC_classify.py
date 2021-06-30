from datetime import datetime

import ee

import pytz

from coastsat.SDS_classify import *
from coastsat import NOC_shoreline
from coastsat.SDS_download import filter_S2_collection, create_folder_structure, download_tif, get_metadata, \
    merge_overlapping_images
from coastsat.SDS_shoreline import calculate_features


def label_images_4classes(metadata, settings):
    """
    Load satellite images and interactively label different classes (hard-coded)

    KV WRL 2019

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'labels': dict
            list of label names (key) and label numbers (value) for each class
        'flood_fill': boolean
            True to use the flood_fill functionality when labelling sand pixels
        'tolerance': float
            tolerance value for flood fill when labelling the sand pixels
        'filepath_train': str
            directory in which to save the labelled data
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)

    Returns:
    -----------
    Stores the labelled data in the specified directory

    """

    filepath_train = settings['filepath_train']
    # initialize figure
    fig, ax = plt.subplots(1, 1, figsize=[17, 10], tight_layout=True, sharex=True,
                           sharey=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'], satname)
        filenames = metadata[satname]['filenames']
        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # read and preprocess image
            image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = SDS_preprocess.preprocess_single(fn, satname,
                                                                                                     settings[
                                                                                                         'cloud_mask_issue'])
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh'] or cloud_cover == 1:
                continue
            # get individual RGB image
            image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
            image_NDVI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 2], cloud_mask)
            image_NDWI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)
            # initialise labels
            image_viz = image_RGB.copy()
            image_labels = np.zeros([image_RGB.shape[0], image_RGB.shape[1]])
            # show RGB image
            ax.axis('off')
            ax.imshow(image_RGB)
            implot = ax.imshow(image_viz, alpha=0.6)
            filename = filenames[i][:filenames[i].find('.')][:-4]
            ax.set_title(filename)

            ##############################################################
            # select image to label
            ##############################################################
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            key_event = {}

            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key

            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                  transform=ax.transAxes,
                                  bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()

                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled labelling images')
                else:
                    plt.waitforbuttonpress()

            # if user decided to skip show the next image
            if skip_image:
                ax.clear()
                continue
            # otherwise label this image
            else:
                ##############################################################
                # digitize white-water pixels
                ##############################################################
                color_ww = settings['colors']['white-water']
                ax.set_title('Click on individual WHITE-WATER pixels (no flood fill)\nwhen finished press <Enter>')
                # create erase button, if you click there it deletes the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                ww_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last labelled pixels
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(ww_pixels) > 0:
                            image_labels[ww_pixels[-1][1], ww_pixels[-1][0]] = 3
                            for k in range(image_viz.shape[2]):
                                image_viz[ww_pixels[-1][1], ww_pixels[-1][0], k] = \
                                                        image_RGB[ww_pixels[-1][1], ww_pixels[-1][0], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del ww_pixels[-1]
                    else:

                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_ww = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_ww] = settings['labels']['white-water']
                        ww_pixels.append(fill_ww)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['white-water'], k] = color_ww[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww = image_viz.copy()
                btn_erase.set(text='<Esc> to Erase', fontsize=12)

                ##############################################################
                # digitize water pixels (with lassos)
                ##############################################################
                color_water = settings['colors']['water']
                ax.set_title('Click and hold to draw lassos and select WATER pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_water = SelectFromImage(ax, implot, color_water)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_water.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_water.array = image_ww
                        implot.set_data(selector_water.array)
                        fig.canvas.draw_idle()
                        selector_water.implot = implot
                        selector_water.image_bool = np.zeros(
                            (selector_water.array.shape[0], selector_water.array.shape[1]))
                        selector_water.ind = []
                        # update image_viz and image_labels
                image_viz = selector_water.array
                selector_water.image_bool = selector_water.image_bool.astype(bool)
                image_labels[selector_water.image_bool] = settings['labels']['water']

                ##############################################################
                # digitize land_1 pixels
                ##############################################################
                ax.set_title(
                    'Click on LAND_1 pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>' %
                    settings['tolerance'])
                # create erase button, if you click there it delets the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                color_land_1 = settings['colors']['land_1']
                land_1_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last selection
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(land_1_pixels) > 0:
                            image_labels[land_1_pixels[-1]] = 0
                            for k in range(image_viz.shape[2]):
                                image_viz[land_1_pixels[-1], k] = image_RGB[land_1_pixels[-1], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del land_1_pixels[-1]

                    # otherwise label the selected land_1 pixels
                    else:
                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_land = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_land] = settings['labels']['land_1']
                        land_1_pixels.append(fill_land)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['land_1'], k] = color_land_1[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww_water_land1 = image_viz.copy()

                ##############################################################
                # digitize land pixels (with lassos)
                ##############################################################
                color_land_2 = settings['colors']['land_2']
                ax.set_title('Click and hold to draw lassos and select LAND_2 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_2)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_viz and image_labels
                image_viz = selector_land.array
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_2']

                image_ww_water_land1_land2 = image_viz.copy()

                ##############################################################
                # digitize land pixels (with lassos)
                ##############################################################
                color_land_3 = settings['colors']['land_3']
                ax.set_title('Click and hold to draw lassos and select LAND_3 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_3)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1_land2
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_labels
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_3']

                # save labelled image
                ax.set_title(filename)
                fig.canvas.draw_idle()
                fp = os.path.join(filepath_train, settings['inputs']['sitename'])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fig.savefig(os.path.join(fp, filename + '.jpg'), dpi=150)
                ax.clear()
                # save labels and features
                features = dict([])
                for key in settings['labels'].keys():
                    image_bool = image_labels == settings['labels'][key]
                    features[key] = SDS_shoreline.calculate_features(image_ms, cloud_mask, image_bool)
                training_data = {'labels': image_labels, 'features': features, 'label_ids': settings['labels']}
                with open(os.path.join(fp, filename + '.pkl'), 'wb') as f:
                    pickle.dump(training_data, f)

    # close figure when finished
    plt.close(fig)

def label_images_5classes(metadata, settings):
    """
    Load satellite images and interactively label different classes (hard-coded)

    KV WRL 2019

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'labels': dict
            list of label names (key) and label numbers (value) for each class
        'flood_fill': boolean
            True to use the flood_fill functionality when labelling sand pixels
        'tolerance': float
            tolerance value for flood fill when labelling the sand pixels
        'filepath_train': str
            directory in which to save the labelled data
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)

    Returns:
    -----------
    Stores the labelled data in the specified directory

    """

    filepath_train = settings['filepath_train']
    # initialize figure
    fig, ax = plt.subplots(1, 1, figsize=[17, 10], tight_layout=True, sharex=True,
                           sharey=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'], satname)
        filenames = metadata[satname]['filenames']
        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # read and preprocess image
            image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = SDS_preprocess.preprocess_single(fn, satname,
                                                                                                     settings[
                                                                                                         'cloud_mask_issue'])
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh'] or cloud_cover == 1:
                continue
            # get individual RGB image
            image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
            image_NDVI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 2], cloud_mask)
            image_NDWI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)
            # initialise labels
            image_viz = image_RGB.copy()
            image_labels = np.zeros([image_RGB.shape[0], image_RGB.shape[1]])
            # show RGB image
            ax.axis('off')
            ax.imshow(image_RGB)
            implot = ax.imshow(image_viz, alpha=0.6)
            filename = filenames[i][:filenames[i].find('.')][:-4]
            ax.set_title(filename)

            ##############################################################
            # select image to label
            ##############################################################
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            key_event = {}

            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key

            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                  transform=ax.transAxes,
                                  bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()

                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled labelling images')
                else:
                    plt.waitforbuttonpress()

            # if user decided to skip show the next image
            if skip_image:
                ax.clear()
                continue
            # otherwise label this image
            else:
                ##############################################################
                # digitize white-water pixels
                ##############################################################
                color_ww = settings['colors']['white-water']
                ax.set_title('Click on individual WHITE-WATER pixels (no flood fill)\nwhen finished press <Enter>')
                # create erase button, if you click there it deletes the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                ww_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last labelled pixels
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(ww_pixels) > 0:
                            image_labels[ww_pixels[-1][1], ww_pixels[-1][0]] = 3
                            for k in range(image_viz.shape[2]):
                                image_viz[ww_pixels[-1][1], ww_pixels[-1][0], k] = \
                                                        image_RGB[ww_pixels[-1][1], ww_pixels[-1][0], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del ww_pixels[-1]
                    else:

                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_ww = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_ww] = settings['labels']['white-water']
                        ww_pixels.append(fill_ww)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['white-water'], k] = color_ww[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww = image_viz.copy()
                btn_erase.set(text='<Esc> to Erase', fontsize=12)

                ##############################################################
                # digitize water pixels (with lassos)
                ##############################################################
                color_water = settings['colors']['water']
                ax.set_title('Click and hold to draw lassos and select WATER pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_water = SelectFromImage(ax, implot, color_water)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_water.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_water.array = image_ww
                        implot.set_data(selector_water.array)
                        fig.canvas.draw_idle()
                        selector_water.implot = implot
                        selector_water.image_bool = np.zeros(
                            (selector_water.array.shape[0], selector_water.array.shape[1]))
                        selector_water.ind = []
                        # update image_viz and image_labels
                image_viz = selector_water.array
                selector_water.image_bool = selector_water.image_bool.astype(bool)
                image_labels[selector_water.image_bool] = settings['labels']['water']

                ##############################################################
                # digitize land_1 pixels
                ##############################################################
                ax.set_title(
                    'Click on LAND_1 pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>' %
                    settings['tolerance'])
                # create erase button, if you click there it delets the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                color_land_1 = settings['colors']['land_1']
                land_1_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last selection
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(land_1_pixels) > 0:
                            image_labels[land_1_pixels[-1]] = 0
                            for k in range(image_viz.shape[2]):
                                image_viz[land_1_pixels[-1], k] = image_RGB[land_1_pixels[-1], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del land_1_pixels[-1]

                    # otherwise label the selected land_1 pixels
                    else:
                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_land = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_land] = settings['labels']['land_1']
                        land_1_pixels.append(fill_land)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['land_1'], k] = color_land_1[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww_water_land1 = image_viz.copy()

                ##############################################################
                # digitize land pixels (with lassos)
                ##############################################################
                color_land_2 = settings['colors']['land_2']
                ax.set_title('Click and hold to draw lassos and select LAND_2 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_2)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_viz and image_labels
                image_viz = selector_land.array
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_2']

                image_ww_water_land1_land2 = image_viz.copy()

                ##############################################################
                # digitize land pixels (with lassos)
                ##############################################################
                color_land_3 = settings['colors']['land_3']
                ax.set_title('Click and hold to draw lassos and select LAND_3 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_3)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1_land2
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_labels
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_3']

                # save labelled image
                ax.set_title(filename)
                fig.canvas.draw_idle()
                fp = os.path.join(filepath_train, settings['inputs']['sitename'])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fig.savefig(os.path.join(fp, filename + '.jpg'), dpi=150)
                ax.clear()
                # save labels and features
                features = dict([])
                for key in settings['labels'].keys():
                    image_bool = image_labels == settings['labels'][key]
                    features[key] = SDS_shoreline.calculate_features(image_ms, cloud_mask, image_bool)
                training_data = {'labels': image_labels, 'features': features, 'label_ids': settings['labels']}
                with open(os.path.join(fp, filename + '.pkl'), 'wb') as f:
                    pickle.dump(training_data, f)

    # close figure when finished
    plt.close(fig)

def label_images_6classes(metadata, settings):


    filepath_train = settings['filepath_train']
    # initialize figure
    fig, ax = plt.subplots(1, 1, figsize=[17, 10], tight_layout=True, sharex=True,
                           sharey=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'], satname)
        filenames = metadata[satname]['filenames']
        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # read and preprocess image
            image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = SDS_preprocess.preprocess_single(fn, satname,
                                                                                                     settings[
                                                                                                         'cloud_mask_issue'])
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh'] or cloud_cover == 1:
                continue
            # get individual RGB image
            image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
            image_NDVI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 2], cloud_mask)
            image_NDWI = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)
            # initialise labels
            image_viz = image_RGB.copy()
            image_labels = np.zeros([image_RGB.shape[0], image_RGB.shape[1]])
            # show RGB image
            ax.axis('off')
            ax.imshow(image_RGB)
            implot = ax.imshow(image_viz, alpha=0.6)
            filename = filenames[i][:filenames[i].find('.')][:-4]
            ax.set_title(filename)

            ##############################################################
            # select image to label
            ##############################################################
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            key_event = {}

            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key

            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                  transform=ax.transAxes,
                                  bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()

                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled labelling images')
                else:
                    plt.waitforbuttonpress()

            # if user decided to skip show the next image
            if skip_image:
                ax.clear()
                continue
            # otherwise label this image
            else:
                ##############################################################
                # digitize white-water pixels
                ##############################################################
                color_ww = settings['colors']['white-water']
                ax.set_title('Click on individual WHITE-WATER pixels (no flood fill)\nwhen finished press <Enter>')
                # create erase button, if you click there it deletes the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                ww_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last labelled pixels
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(ww_pixels) > 0:
                            image_labels[ww_pixels[-1][1], ww_pixels[-1][0]] = 3
                            for k in range(image_viz.shape[2]):
                                image_viz[ww_pixels[-1][1], ww_pixels[-1][0], k] = \
                                                        image_RGB[ww_pixels[-1][1], ww_pixels[-1][0], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del ww_pixels[-1]
                    else:

                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_ww = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_ww] = settings['labels']['white-water']
                        ww_pixels.append(fill_ww)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['white-water'], k] = color_ww[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww = image_viz.copy()
                btn_erase.set(text='<Esc> to Erase', fontsize=12)

                ##############################################################
                # digitize water pixels (with lassos)
                ##############################################################
                color_water = settings['colors']['water']
                ax.set_title('Click and hold to draw lassos and select WATER pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_water = SelectFromImage(ax, implot, color_water)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_water.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_water.array = image_ww
                        implot.set_data(selector_water.array)
                        fig.canvas.draw_idle()
                        selector_water.implot = implot
                        selector_water.image_bool = np.zeros(
                            (selector_water.array.shape[0], selector_water.array.shape[1]))
                        selector_water.ind = []
                        # update image_viz and image_labels
                image_viz = selector_water.array
                selector_water.image_bool = selector_water.image_bool.astype(bool)
                image_labels[selector_water.image_bool] = settings['labels']['water']

                ##############################################################
                # digitize land_1 pixels - hard surfaces
                ##############################################################
                ax.set_title(
                    'Click on LAND_1 pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>' %
                    settings['tolerance'])
                # create erase button, if you click there it deletes the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                color_land_1 = settings['colors']['land_1']
                land_1_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last selection
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(land_1_pixels) > 0:
                            image_labels[land_1_pixels[-1]] = 0
                            for k in range(image_viz.shape[2]):
                                image_viz[land_1_pixels[-1], k] = image_RGB[land_1_pixels[-1], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del land_1_pixels[-1]

                    # otherwise label the selected land_1 pixels
                    else:
                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_land = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_land] = settings['labels']['land_1']
                        land_1_pixels.append(fill_land)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['land_1'], k] = color_land_1[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()

                image_ww_water_land1 = image_viz.copy()

                ##############################################################
                # digitize land_2 pixels (with lassos) - nature
                ##############################################################
                color_land_2 = settings['colors']['land_2']
                ax.set_title('Click and hold to draw lassos and select LAND_2 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_2)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_viz and image_labels
                image_viz = selector_land.array
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_2']

                image_ww_water_land1_land2 = image_viz.copy()

                ##############################################################
                # digitize land_3 pixels (with lassos) - urban
                ##############################################################
                color_land_3 = settings['colors']['land_3']
                ax.set_title('Click and hold to draw lassos and select LAND_3 pixels\nwhen finished press <Enter>')
                fig.canvas.draw_idle()
                selector_land = SelectFromImage(ax, implot, color_land_3)
                key_event = {}
                while True:
                    fig.canvas.draw_idle()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        selector_land.disconnect()
                        break
                    elif key_event.get('pressed') == 'escape':
                        selector_land.array = image_ww_water_land1_land2
                        implot.set_data(selector_land.array)
                        fig.canvas.draw_idle()
                        selector_land.implot = implot
                        selector_land.image_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1]))
                        selector_land.ind = []
                # update image_labels
                selector_land.image_bool = selector_land.image_bool.astype(bool)
                image_labels[selector_land.image_bool] = settings['labels']['land_3']

                image_ww_water_land1_land2_land3 = image_viz.copy()

                ##############################################################
                # digitize sand pixels
                ##############################################################
                ax.set_title(
                    'Click on SAND pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>' %
                    settings['tolerance'])
                # create erase button, if you click there it deletes the last selection
                btn_erase = ax.text(image_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                fig.canvas.draw_idle()
                color_sand = settings['colors']['sand']
                sand_pixels = []
                while 1:
                    seed = ginput(n=1, timeout=0, show_clicks=True)
                    # if empty break the loop and go to next label
                    if len(seed) == 0:
                        break
                    else:
                        # round to pixel location
                        seed = np.round(seed[0]).astype(int)
                        # if user clicks on erase, delete the last selection
                    if seed[0] > 0.95 * image_ms.shape[1] and seed[1] < 0.05 * image_ms.shape[0]:
                        if len(sand_pixels) > 0:
                            image_labels[sand_pixels[-1]] = 0
                            for k in range(image_viz.shape[2]):
                                image_viz[sand_pixels[-1], k] = image_RGB[sand_pixels[-1], k]
                            implot.set_data(image_viz)
                            fig.canvas.draw_idle()
                            del sand_pixels[-1]

                    # otherwise label the selected land_1 pixels
                    else:
                        # flood fill the NDVI and the NDWI
                        fill_NDVI = flood(image_NDVI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        fill_NDWI = flood(image_NDWI, (seed[1], seed[0]), tolerance=settings['tolerance'])
                        # compute the intersection of the two masks
                        fill_sand = np.logical_and(fill_NDVI, fill_NDWI)
                        image_labels[fill_sand] = settings['labels']['sand']
                        sand_pixels.append(fill_sand)
                        # show the labelled pixels
                        for k in range(image_viz.shape[2]):
                            image_viz[image_labels == settings['labels']['sand'], k] = color_sand[k]
                        implot.set_data(image_viz)
                        fig.canvas.draw_idle()


                # save labelled image
                ax.set_title(filename)
                fig.canvas.draw_idle()
                fp = os.path.join(filepath_train, settings['inputs']['sitename'])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                fig.savefig(os.path.join(fp, filename + '.jpg'), dpi=150)
                ax.clear()
                # save labels and features
                features = dict([])
                for key in settings['labels'].keys():
                    image_bool = image_labels == settings['labels'][key]
                    features[key] = SDS_shoreline.calculate_features(image_ms, cloud_mask, image_bool)
                training_data = {'labels': image_labels, 'features': features, 'label_ids': settings['labels']}
                with open(os.path.join(fp, filename + '.pkl'), 'wb') as f:
                    pickle.dump(training_data, f)

    # close figure when finished
    plt.close(fig)


def classify_image_NN(image_ms, classes, cloud_mask, min_beach_area, classifier):

    # calculate features
    vec_features = calculate_features(image_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = classifier.predict(vec_features)

    # recompose image
    vec_classifier = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classifier[~vec_mask] = labels
    image_classifier = vec_classifier.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    images_layers = []
    for key in classes.keys():

        class_label = classes[key][0]
        layer = image_classifier == class_label
        layer = morphology.remove_small_objects(layer, min_size=min_beach_area, connectivity=2)

        images_layers.append(layer)

    image_labels = np.stack(images_layers, axis=-1)

    return image_classifier, image_labels


def evaluate_classifier(classifier, metadata, settings, base_path):
    # create folder called evaluation
    fp = os.path.join(base_path, 'evaluation')
    if not os.path.exists(fp):
        os.makedirs(fp)

    # initialize figure (not interactive)
    plt.ioff()
    fig, ax = plt.subplots(1, 2, figsize=[17, 10], sharex=True, sharey=True,
                           constrained_layout=True)

    # loop through satellites
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings['inputs'], satname)
        filenames = metadata[satname]['filenames']

        # load classifiers and
        if satname in ['L5', 'L7', 'L8']:
            pixel_size = 15
        elif satname == 'S2':
            pixel_size = 10
        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels

        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)

        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # read and preprocess image
            image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = \
                SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])

            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            image_classif, image_labels = NOC_shoreline.classify_image_NN_6classes(image_ms, cloud_mask,
                                                                                   min_beach_area_pixels, classifier)

            # make a plot
            image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
            # create classified image
            image_class = np.copy(image_RGB)

            # for each class add calssified colours to image_class
            classes = settings['classes']
            class_keys = classes.keys()

            for key in class_keys:
                class_label = classes[key][0]
                class_colour = classes[key][1]
                image_class[image_labels[:, :, class_label - 1], 0] = class_colour[0]
                image_class[image_labels[:, :, class_label - 1], 1] = class_colour[1]
                image_class[image_labels[:, :, class_label - 1], 2] = class_colour[2]

            # show images
            ax[0].imshow(image_RGB)
            #            ax[1].imshow(image_RGB)
            ax[1].imshow(image_class, alpha=0.75)
            ax[0].axis('off')
            ax[1].axis('off')
            filename = filenames[i][:filenames[i].find('.')][:-4]
            ax[0].set_title(filename)
            # save figure
            fig.savefig(os.path.join(fp, settings['inputs']['sitename'] + filename[:19] + '.jpg'), dpi=150)
            # clear axes
            for cax in fig.axes:
                cax.clear()

    # close the figure at the end
    plt.close()


def retrieve_training_images(inputs):

    # initialise connection with GEE server
    ee.Initialize()

    # check image availabiliy and retrieve list of images
    image_dict_T1 = check_training_images_available(inputs)

    # remove UTM duplicates in S2 collections (they provide several projections for same images)
    if 'S2' in inputs['sat_list'] and len(image_dict_T1['S2']) > 0:
        image_dict_T1['S2'] = filter_S2_collection(image_dict_T1['S2'])

    # create a new directory for this site with the name of the site
    median_dir_path = os.path.join(inputs['filepath'], inputs['sitename'])
    if not os.path.exists(median_dir_path): os.makedirs(median_dir_path)

    print('\nDownloading images:')
    suffix = '.tif'
    for sat_name in image_dict_T1.keys():
        print('%s: %d images' % (sat_name, len(image_dict_T1[sat_name])))
        # create subdir structure to store the different bands
        filepaths = create_folder_structure(median_dir_path, sat_name)
        # initialise variables and loop through images
        georef_accs = []
        filenames = []
        all_names = []
        image_epsg = []
        for i in range(5):

            image_meta = image_dict_T1[sat_name][i]

            # get time of acquisition (UNIX time) and convert to datetime
            t = image_meta['properties']['system:time_start']
            image_timestamp = datetime.fromtimestamp(t / 1000, tz=pytz.utc)
            image_date = image_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

            # get epsg code
            image_epsg.append(int(image_meta['bands'][0]['crs'][5:]))

            # get geometric accuracy
            if sat_name in ['L5', 'L7', 'L8']:
                if 'GEOMETRIC_RMSE_MODEL' in image_meta['properties'].keys():
                    acc_georef = image_meta['properties']['GEOMETRIC_RMSE_MODEL']
                else:
                    acc_georef = 12  # default value of accuracy (RMSE = 12m)
            elif sat_name in ['S2']:
                # Sentinel-2 products don't provide a georeferencing accuracy (RMSE as in Landsat)
                # but they have a flag indicating if the geometric quality control was passed or failed
                # if passed a value of 1 is stored if failed a value of -1 is stored in the metadata
                # the name of the property containing the flag changes across the S2 archive
                # check which flag name is used for the image and store the 1/-1 for acc_georef
                flag_names = ['GEOMETRIC_QUALITY_FLAG', 'GEOMETRIC_QUALITY', 'quality_check']
                for key in flag_names:
                    if key in image_meta['properties'].keys(): break
                if image_meta['properties'][key] == 'PASSED':
                    acc_georef = 1
                else:
                    acc_georef = -1
            georef_accs.append(acc_georef)

            bands = {}
            image_filename = {}
            # first delete dimensions key from dictionary
            # otherwise the entire image is extracted (don't know why)
            image_bands = image_meta['bands']
            for j in range(len(image_bands)): del image_bands[j]['dimensions']

            # Landsat 5 download
            if sat_name == 'L5':
                bands[''] = [image_bands[0], image_bands[1], image_bands[2], image_bands[3],
                             image_bands[4], image_bands[7]]
                image_filename[''] = image_date + '_' + sat_name + '_' + inputs['sitename'] + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(image_filename[''] in _ for _ in all_names):
                    image_filename[''] = image_date + '_' + sat_name + '_' + inputs['sitename'] + '_dup' + suffix
                all_names.append(image_filename[''])
                filenames.append(image_filename[''])
                # download .tif from EE
                while True:
                    try:
                        image_ee = ee.Image(image_meta['id'])
                        local_data = download_tif(image_ee, inputs['polygon'], bands[''], filepaths[1])
                        break
                    except:
                        continue
                # rename the file as the image is downloaded as 'data.tif'
                try:
                    os.rename(local_data, os.path.join(filepaths[1], image_filename['']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], image_filename['']))
                    os.rename(local_data, os.path.join(filepaths[1], image_filename['']))
                # metadata for .txt file
                txt_file_name = image_filename[''].replace('.tif', '')
                metadict = {'filename': image_filename[''], 'acc_georef': georef_accs[i],
                            'epsg': image_epsg[i]}

            # Landsat 7 and 8 download
            elif sat_name in ['L7', 'L8']:
                if sat_name == 'L7':
                    bands['pan'] = [image_bands[8]]  # panchromatic band
                    bands['ms'] = [image_bands[0], image_bands[1], image_bands[2], image_bands[3],
                                   image_bands[4], image_bands[9]]  # multispectral bands
                else:
                    bands['pan'] = [image_bands[7]]  # panchromatic band
                    bands['ms'] = [image_bands[1], image_bands[2], image_bands[3], image_bands[4],
                                   image_bands[5], image_bands[11]]  # multispectral bands
                for key in bands.keys():
                    image_filename[key] = image_date + '_' + sat_name + '_' + inputs['sitename'] + '_' + key + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(image_filename['pan'] in _ for _ in all_names):
                    for key in bands.keys():
                        image_filename[key] = image_date + '_' + sat_name + '_' + inputs['sitename'] + '_' + key + '_dup' + suffix
                all_names.append(image_filename['pan'])
                filenames.append(image_filename['pan'])
                # download .tif from EE (panchromatic band and multispectral bands)
                while True:
                    try:
                        image_ee = ee.Image(image_meta['id'])
                        local_data_pan = download_tif(image_ee, inputs['polygon'], bands['pan'], filepaths[1])
                        local_data_ms = download_tif(image_ee, inputs['polygon'], bands['ms'], filepaths[2])
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                try:  # panchromatic
                    os.rename(local_data_pan, os.path.join(filepaths[1], image_filename['pan']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], image_filename['pan']))
                    os.rename(local_data_pan, os.path.join(filepaths[1], image_filename['pan']))
                try:  # multispectral
                    os.rename(local_data_ms, os.path.join(filepaths[2], image_filename['ms']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], image_filename['ms']))
                    os.rename(local_data_ms, os.path.join(filepaths[2], image_filename['ms']))
                # metadata for .txt file
                base_file_name = image_filename['pan'].replace('_pan', '')
                txt_file_name = base_file_name.replace('.tif', '.txt')
                metadict = {'filename': base_file_name, 'acc_georef': georef_accs[i],
                            'epsg': image_epsg[i]}

            # Sentinel-2 download
            elif sat_name in ['S2']:

                bands['10m'] = [image_bands[1], image_bands[2], image_bands[3], image_bands[7]]  # multispectral bands
                bands['20m'] = [image_bands[11]]  # SWIR band
                bands['60m'] = [image_bands[15]]  # QA band
                for key in bands.keys():
                    image_filename[key] = image_date + '_' + sat_name + '_' + inputs['sitename'] + '_' + key + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(image_filename['10m'] in _ for _ in all_names):
                    for key in bands.keys():
                        image_filename[key] = image_date + '_' + sat_name + '_' + inputs['sitename'] + '_' + key + '_dup' + suffix
                    # also check for triplicates (only on S2 imagery) and add 'tri' to the name
                    if image_filename['10m'] in all_names:
                        for key in bands.keys():
                            image_filename[key] = image_date + '_' + sat_name + '_' + inputs[
                                'sitename'] + '_' + key + '_tri' + suffix
                all_names.append(image_filename['10m'])
                filenames.append(image_filename['10m'])

                # download .tif from EE (multispectral bands at 3 different resolutions)
                while True:
                    try:
                        image_ee = ee.Image(image_meta['id'])
                        local_data_10m = download_tif(image_ee, inputs['polygon'], bands['10m'], filepaths[1])
                        local_data_20m = download_tif(image_ee, inputs['polygon'], bands['20m'], filepaths[2])
                        local_data_60m = download_tif(image_ee, inputs['polygon'], bands['60m'], filepaths[3])
                        break
                    except:
                        continue

                # rename the files as the image is downloaded as 'data.tif'
                try:  # 10m
                    os.rename(local_data_10m, os.path.join(filepaths[1], image_filename['10m']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], image_filename['10m']))
                    os.rename(local_data_10m, os.path.join(filepaths[1], image_filename['10m']))
                try:  # 20m
                    os.rename(local_data_20m, os.path.join(filepaths[2], image_filename['20m']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], image_filename['20m']))
                    os.rename(local_data_20m, os.path.join(filepaths[2], image_filename['20m']))
                try:  # 60m
                    os.rename(local_data_60m, os.path.join(filepaths[3], image_filename['60m']))
                except:  # overwrite if already exists
                    os.remove(os.path.join(filepaths[3], image_filename['60m']))
                    os.rename(local_data_60m, os.path.join(filepaths[3], image_filename['60m']))
                # metadata for .txt file
                base_file_name = image_filename['10m'].replace('_10m', '')
                txt_file_name = base_file_name.replace('.tif', '.txt')
                metadict = {'filename': base_file_name, 'acc_georef': georef_accs[i],
                            'epsg': image_epsg[i]}

            # write metadata
            with open(os.path.join(base_file_name, txt_file_name), 'w') as f:
                for key in metadict.keys():
                    f.write('%s\t%s\n' % (key, metadict[key]))
            # print percentage completion for user

    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs)

    # merge overlapping images (necessary only if the polygon is at the boundary of an image)
    if 'S2' in metadata.keys():
        try:
            metadata = merge_overlapping_images(metadata, inputs)
        except:
            print('WARNING: there was an error while merging overlapping S2 images,' +
                  ' please open an issue on Github at https://github.com/kvos/CoastSat/issues' +
                  ' and include your script so we can find out what happened.')

    # save metadata dict
    with open(os.path.join(median_dir_path, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return metadata


def check_training_images_available(inputs):

    date_start = inputs['dates'][0]
    date_end = inputs['dates'][1]

    # check if dates are in correct order
    dates = [datetime.strptime(_,'%Y-%m-%d') for _ in inputs['dates']]
    if date_end <= date_start:
        raise Exception('Verify that your dates are in the correct order')

    # check if EE was initialised or not
    try:
        ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')
    except:
        ee.Initialize()

    # check how many images are available in Tier 1 and Sentinel Level-1C
    col_names_T1 = {'L5':'LANDSAT/LT05/C01/T1_TOA',
                 'L7':'LANDSAT/LE07/C01/T1_TOA',
                 'L8':'LANDSAT/LC08/C01/T1_TOA',
                 'S2':'COPERNICUS/S2'}

    image_dict_T1 = {}
    for sat_name in inputs['sat_list']:

        # get list of images in EE collection
        while True:
            try:

                ee_col = ee.ImageCollection(col_names_T1[sat_name])\
                            .filterBounds(ee.Geometry.Polygon(inputs['polygon'])) \
                            .filterDate(inputs['dates'][0],inputs['dates'][1]) \
                            .sort('CLOUDY_PIXEL_PERCENTAGE')

                image_list = ee_col.getInfo().get('features')
                break
            except:
                continue

        image_dict_T1[sat_name] = image_list

    return image_dict_T1