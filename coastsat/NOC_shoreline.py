from scipy.ndimage import gaussian_filter

from coastsat.SDS_shoreline import *

from coastsat import NOC_preprocess, NOC_tools, NOC_classify


def extract_shorelines_optical(metadata, settings):

    inputs = settings['inputs']
    sitename = inputs['sitename']
    classes = settings['classes']

    filepath_data = inputs['filepath']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')

    # initialise output structure
    output = {}
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    # loop through satellite list
    for satname in metadata.keys():

        # get images
        filepath = SDS_tools.get_filepath(inputs, satname)
        filenames = metadata[satname]['filenames']

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = []  # filename of the images from which the shorelines where derived
        output_cloudcover = []  # cloud cover of the images
        output_geoaccuracy = []  # georeferencing accuracy of the images
        output_idxkeep = []  # index that were kept during the analysis (cloudy images are skipped)
        output_median_no = []

        # load classifiers
        if satname in ['L5', 'L7', 'L8']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                classifier = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark.pkl'))
            elif settings['sand_color'] == 'bright':
                classifier = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright.pkl'))
            else:
                classifier = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat.pkl'))

        elif satname == 'S2':
            pixel_size = 10
            classifier = joblib.load(os.path.join(filepath_models, 'NN_6classes_S2.pkl'))

        # convert settings['min_beach_area'] from  metres to pixels
        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)

        print(f'{satname} extracting shorelines for: {len(filenames)} images')
        print()

        # loop through the images
        for file_index in range(len(filenames)):

            # get image filename
            fn = SDS_tools.get_filenames(filenames[file_index], filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = \
                SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][file_index]

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                             (cloud_mask.shape[0] * cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
                continue
            # remove no data pixels from the cloud mask
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, image_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # calculate a buffer around the reference shoreline (if any has been digitised)
            image_ref_buffer = create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                                    pixel_size, settings)

            # classify image with NN classifier
            image_classifier, image_labels = NOC_classify.classify_image_NN(image_ms, classes, cloud_mask,
                                                               min_beach_area_pixels, classifier)

            # find the shoreline interactively
            date = filenames[file_index][:19]
            skip_image, shoreline = adjust_detection_optical(image_ms, cloud_mask, image_labels,
                                                     image_ref_buffer, image_epsg, georef, settings, date,
                                                     satname)
            # if the user decides to skip the image, continue and do not save the mapped shoreline
            if skip_image:
                continue

            # append to output variables
            output_timestamp.append(metadata[satname]['dates'][file_index])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[file_index])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][file_index])
            output_idxkeep.append(file_index)
            output_median_no.append(metadata[satname]['median_no'][file_index])

        # create dictionnary of output
        output[satname] = {
            'dates': output_timestamp,
            'shorelines': output_shoreline,
            'filename': output_filename,
            'cloud_cover': output_cloudcover,
            'geoaccuracy': output_geoaccuracy,
            'idx': output_idxkeep,
            'median_no': output_median_no
        }
        print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output


def find_contours_optical(image_ms, image_labels, cloud_mask, ref_shoreline_buffer):

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    image_mwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    image_wi = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)
    # stack indices together
    image_ind = np.stack((image_wi, image_mwi), axis=-1)
    vec_ind = image_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_land1 = image_labels[:, :, 0].reshape(ncols * nrows)
    vec_land2 = image_labels[:, :, 1].reshape(ncols * nrows)
    vec_land3 = image_labels[:, :, 2].reshape(ncols * nrows)
    vec_sand = image_labels[:, :, 5].reshape(ncols * nrows)
    vec_all_land = np.logical_or(np.logical_or((np.logical_or(vec_land1, vec_land2)),
                                               vec_land3), vec_sand)

    vec_water = image_labels[:, :, 4].reshape(ncols * nrows)
    vec_image_ref_buffer = ref_shoreline_buffer.reshape(ncols * nrows)

    # select land and water pixels that are within the buffer
    int_land = vec_ind[np.logical_and(vec_image_ref_buffer,vec_all_land),:]
    int_sea = vec_ind[np.logical_and(vec_image_ref_buffer,vec_water),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_land) > 0 and len(int_sea) > 0:
        if np.argmin([int_land.shape[0],int_sea.shape[0]]) == 1:
            int_land = int_land[np.random.choice(int_land.shape[0],int_sea.shape[0], replace=False),:]
        else:
            int_sea = int_sea[np.random.choice(int_sea.shape[0],int_land.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_land,int_sea, axis=0)
    int_all = int_all[~np.isnan(int_all)]
    t_mwi = filters.threshold_otsu(int_all)

    # find contour with MS algorithm
    image_mwi_buffer = np.copy(image_mwi)
    image_mwi_buffer[~ref_shoreline_buffer] = np.nan
    contours_mwi = measure.find_contours(image_mwi_buffer, level=t_mwi, mask=ref_shoreline_buffer)
    # remove contour points that are NaNs (around clouds)
    contours_mwi = process_contours(contours_mwi)

    return contours_mwi, t_mwi


def adjust_detection_optical(image_ms, cloud_mask, image_labels, image_ref_buffer, image_epsg, georef,
                     settings, date, satname):

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']

    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    # format date
    date_string = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d')

    #  image_classifiery will become filled with labels
    image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
    image_classifieried = np.copy(image_RGB)

    # compute MNDWI grayscale image
    image_mndwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    image_mndwi_buffer = np.copy(image_mndwi)
    image_mndwi_buffer[~image_ref_buffer] = np.nan

    # for each class add it to image_classifieried and
    # extract the MDWI values for all pixels in that class
    classes = settings['classes']
    class_keys = classes.keys()

    mndwi_pixels = {}
    for key in class_keys:
        class_label = classes[key][0]
        class_colour = classes[key][1]

        image_classifieried[image_labels[:, :, class_label-1], 0] = class_colour[0]
        image_classifieried[image_labels[:, :, class_label-1], 1] = class_colour[1]
        image_classifieried[image_labels[:, :, class_label-1], 2] = class_colour[2]

        mndwi_pixels[class_label] = image_mndwi[image_labels[:, :, class_label-1]]

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    image_RGB = np.where(np.isnan(image_RGB), nan_color, image_RGB)
    image_class = np.where(np.isnan(image_classifieried), 1.0, image_classifieried)

    # plot image 1 (RGB)
    ax1.imshow(image_RGB)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # plot image 2 (classification)
    ax2.imshow(image_class)
    ax2.axis('off')

    patches = [mpatches.Patch(color=[1,1,1], label='unclassified')]

    for key in class_keys:
        class_colour = classes[key][1]
        patches.append(mpatches.Patch(color=class_colour, label=key))

    patches.append(mlines.Line2D([],[],color='k',linestyle='-', label='shoreline'))
#    ax2.legend(handles=patches, bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title(date_string, fontweight='bold', fontsize=16)

    # plot image 3 (MNDWI)
    ax3.imshow(image_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontweight='bold', fontsize=16)

    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='pixels', yticklabels=[], xlim=[-1, 1])

    for key in class_keys:
        class_label = classes[key][0]
        class_colour = classes[key][1]
        class_pixels = mndwi_pixels[class_label]

        alpha = 0.75
        if key == 'sand':
            alpha = 0.5
        if key in ['hard', 'land_1']:
            alpha = 1.0

        if len(class_pixels) > 0:
            bins = np.arange(np.nanmin(class_pixels), np.nanmax(class_pixels) + binwidth, binwidth)
            ax4.hist(class_pixels, bins=bins, density=True,  color=class_colour, label=key, alpha=alpha)

    # use classification to refine threshold and extract the sand/water interface
    contours_mndwi, t_mndwi = find_contours_optical(image_ms, image_labels, cloud_mask, image_ref_buffer)

    # process the water contours into a shoreline
    shoreline = process_shoreline(contours_mndwi, cloud_mask, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_mndwi, ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # update the threshold value
            t_mndwi = pt[0][0]
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(t_mndwi) >= 1: continue
            # update the plot
            t_line.set_xdata([t_mndwi, t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(image_mndwi_buffer, level=t_mndwi,  mask=image_ref_buffer)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours)
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:, [0, 1]], georef)
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}

    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k', fc='w'))
        plt.draw()
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
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline



def extract_shorelines_sar(metadata, settings):

    inputs = settings['inputs']

    sitename = inputs['sitename']
    base_filepath = inputs['filepath']
    satname = inputs['sat_list'][0]
    pixel_size = inputs['pixel_size']

    # initialise output structure
    output = {}
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(base_filepath, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping shorelines:')

    # get images
    data_path = NOC_tools.get_filepath(settings['inputs'], satname)
    filenames = metadata[satname]['filenames']

    # initialise the output variables
    output_timestamp = []  # datetime at which the image was acquired (UTC time)
    output_shoreline = []  # vector of shoreline points
    output_filename = []  # filename of the images from which the shorelines where derived
    output_geoaccuracy = []  # georeferencing accuracy of the images
    output_median_no = []

    # loop through the images
    for i in range(len(filenames)):

        print('\r%s:   %d%%' % (satname, int(((i + 1) / len(filenames)) * 100)))
        print()

        # get image filename
        filename = filenames[i]

        file_path = os.path.join(data_path, filename)

        sar_image, georef = NOC_preprocess.preprocess_sar(file_path, satname)

        # get image spatial reference system (epsg code) from metadata dict
        image_epsg = metadata[satname]['epsg'][i]

        buffer_shape = (sar_image.shape[0], sar_image.shape[1])

        # calculate a buffer around the reference shoreline (if any has been digitised)
        if settings['reference_shoreline'].any():
            image_ref_buffer = create_shoreline_buffer(buffer_shape, georef, image_epsg,
                                                    pixel_size, settings)
        else:
            image_ref_buffer = np.ones(buffer_shape, dtype=np.bool)

        # find the shoreline interactively
        date = filename[:19]
        skip_image, shorelines = adjust_detection_sar(sar_image, image_ref_buffer, image_epsg, georef,
                                                     settings, date,  satname)

        # if the user decides to skip the image, continue and do not save the mapped shoreline
        if skip_image:
            continue

        # append to output variables
        output_timestamp.append(metadata[satname]['dates'][i])
        output_shoreline.append(shorelines)
        output_filename.append(filenames[i])
        output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
        output_median_no.append(metadata[satname]['median_no'][i])

    # create dictionnary of output
    output[satname] = {
        'dates': output_timestamp,
        'shorelines': output_shoreline,
        'filename': output_filename,
        'geoaccuracy': output_geoaccuracy,
        'median_no': output_median_no
    }
    print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(base_filepath, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output


def adjust_detection_sar(sar_image, image_ref_buffer, image_epsg, georef,
                     settings, date, satname):

    inputs = settings['inputs']
    polarisation = inputs['polarisation']

    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')

    # format date
    date_str = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d')

    # compute classified image
    if polarisation == 'VV':
        image_pol = np.copy(sar_image[:,:,0])
        colour = [1, 0, 1, 1]
    else:
        image_pol = np.copy(sar_image[:,:,1])
        colour = [1, 1, 0, 1]

    if inputs['sigma'] != 0:
        image_pol = gaussian_filter(image_pol, sigma=inputs['sigma'], mode='reflect')

    # and the vectors needed for the histogram
    cols = sar_image.shape[0]
    rows = sar_image.shape[1]
    vec_pol = image_pol.reshape(cols*rows)

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # plot image 1 (RGB)
    ax1.imshow(image_pol)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # plot image 1 (grey)
    ax2.imshow(image_pol, cmap='gray')
    ax2.axis('off')
    ax2.set_title(date_str, fontweight='bold', fontsize=16)

    # plot image 3 (blue/red)
    ax3.imshow(image_pol, cmap='bwr')
    ax3.axis('off')
    ax3.set_title(polarisation, fontweight='bold', fontsize=16)

    # plot histogram of sigma values
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF', yticklabels=[], xlim=[np.nanmin(sar_image), np.nanmax(sar_image)])

    binwidth = 0.1
    bins = np.arange(np.nanmin(vec_pol), np.nanmax(vec_pol) + binwidth, binwidth)
    ax4.hist(vec_pol, bins=bins, density=True, color=colour, label=polarisation)

    t_sar = filters.threshold_otsu(image_pol)
    contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)

    # process the water contours into a shoreline
    shorelines = process_sar_shoreline(contours_sar, georef, image_epsg, settings)

    # convert shoreline to pixels
    if len(shorelines) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shorelines,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        print(f'@@@ no shorelines yet')

        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
    t_line = ax4.axvline(x=t_sar, ls='--', c='r', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')

    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # update the threshold value
            t_sar = pt[0][0]
            # update the plot
            t_line.set_xdata([t_sar, t_sar])
            # map contours with new threshold
            contours_sar = measure.find_contours(image_pol, level=t_sar,
                                                 fully_connected='high', mask=image_ref_buffer)
            # process the water contours into a shoreline
            shorelines = process_sar_shoreline(contours_sar, georef, image_epsg, settings)

            for shoreline in shorelines:
                # convert shoreline to pixels
                if len(shoreline) > 0:
                    sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                                settings['output_epsg'],
                                                                                image_epsg)[:, [0, 1]], georef)
                else:
                    sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                # update the plotted shorelines
                sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('sigma0 pixel intensities and threshold')
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}

    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k', fc='w'))
        plt.draw()
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
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    if inputs['sigma'] != 0:
        filepath = os.path.join(inputs['filepath'], sitename)
        with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'wb') as f:
            pickle.dump(shorelines, f)

    return skip_image, shorelines


def process_sar_shoreline(contours, georef, image_epsg, settings):

    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    contours_array = np.transpose(np.array([x_points, y_points]))

    shoreline = contours_array

    return shoreline
