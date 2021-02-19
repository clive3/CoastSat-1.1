from coastsat.SDS_tools import *

def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.

    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read

    Returns:
    -----------
    polygon: list
        coordinates extracted from the .kml file

    """

    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read()
        # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1) + len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')

    coordlist = subdoc.split(',')
    # read coordinates
    polygon = []

    for coord_pair in range(int(len(coordlist) / 2)):
        lat = float(coordlist[coord_pair * 2].replace('0 ', ''))
        lon = float(coordlist[coord_pair * 2 + 1].replace('0 ', ''))
        polygon.append([lat, lon])

    # read coordinates
    #    polygon = []
    #    for i in range(1, len(coordlist) - 1):
    #        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])

    return [polygon]


def output_to_gdf(output):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame

    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes

    Returns:
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes

    """

    # loop through the mapped shorelines
    counter = 0
    gdf_all = None
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            geom = geometry.LineString(output['shorelines'][i])
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i, 'date'] = output['dates'][i]
            gdf.loc[i, 'satname'] = output['satname'][i]
            gdf.loc[i, 'Median_no'] = output['median_no'][i]

            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1

    return gdf_all


def get_filepath(inputs, satname):
    """
    Create filepath to the different folders containing the satellite images.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')

    Returns:
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images

    """

    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 8 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(filepath_data, sitename, satname, '10m')
        filepath20 = os.path.join(filepath_data, sitename, satname, '20m')
        filepath60 = os.path.join(filepath_data, sitename, satname, '60m')
        filepath = [filepath10, filepath20, filepath60]

    elif satname == 'S1':
        # access downloaded Sentinel 1 images
        filepath = os.path.join(filepath_data, sitename, satname)

    return filepath