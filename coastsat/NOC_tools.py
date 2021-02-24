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


def output_to_gdf_median(output):

    # loop through the mapped shorelines
    gdf_all = None
    for index, shoreline in enumerate(output['shorelines']):
        # skip if there shoreline is empty
        if len(shoreline) == 0:
            continue
        else:
            geom = geometry.LineString(shoreline)
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [index]
            gdf.loc[index, 'date_start'] = output['date_start']
            gdf.loc[index, 'date_end'] = output['date_end']
            gdf.loc[index, 'sat_name'] = output['sat_name']
            gdf.loc[index, 'number_median_images'] = output['number_median_images']

            # store into geodataframe
            if index == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)

    return gdf_all


def merge_output_median(output):

    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['sat_name'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['sat_name'] = output_all['sat_name'] + [_ for _ in np.tile(satname,
                                                                            len(output[satname]['date_start']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['date_start'])), key=output_all['date_start'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all
