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

