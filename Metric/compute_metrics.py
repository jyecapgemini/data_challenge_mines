import glob
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm


def calculate_iou(xmin_pred, xmax_pred, ymin_pred, ymax_pred, xmin_real,
                  xmax_real, ymin_real, ymax_real):
    """This function calculates the IoU between two polygons.

    Args:
        xmin_pred: Value of the prediction min x-axis.
        xmax_pred: Value of the prediction max x-axis.
        ymin_pred: Value of the prediction min y-axis.
        ymax_pred: Value of the prediction max y-axis.
        xmin_real: Value of the real min x-axis.
        xmax_real: Value of the real max x-axis.
        ymin_real: Value of the real min y-axis.
        ymax_real: Value of the real max y-axis.

    Returns:
        The return value is the intersection over union.

    """
    if (xmin_real, xmax_real, ymin_real, ymax_real) == (0, 0, 0, 0):
        if (xmin_pred, xmax_pred, ymin_pred, ymax_pred) == (0, 0, 0, 0):
            return 1

        else:
            return 0

    else:
        box_pred = [[xmin_pred, ymin_pred], [xmax_pred, ymin_pred],
                    [xmax_pred, ymax_pred], [xmin_pred, ymax_pred]]
        box_real = [[xmin_real, ymin_real], [xmax_real, ymin_real],
                    [xmax_real, ymax_real], [xmin_real, ymax_real]]
        poly_1 = Polygon(box_pred)
        poly_2 = Polygon(box_real)
        try:
            iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
            return iou
        except:
            return 0


def get_coordinates(data, ID):
    """This function gets the prediction and real coordinates.

    Args:
        data: Input dataframe.
        ID: Image Id.

    Returns:
        The return value is xmin_pred, xmax_pred, ymin_pred, ymax_pred, xmin_real, xmax_real, ymin_real, ymax_real.

    """
    real_box = data.loc[
        data.im_name == ID,
        ['x_min_real', "y_min_real", "x_max_real", "y_max_real"]].values[0]
    xmin_real, ymin_real = int(real_box[0]), int(real_box[1])
    xmax_real, ymax_real = int(real_box[2]), int(real_box[3])

    pred_box = data.loc[
        data.im_name == ID,
        ['x_min_pred', "y_min_pred", "x_max_pred", "y_max_pred"]].values[0]
    xmin_pred, ymin_pred = int(pred_box[0]), int(pred_box[1])
    xmax_pred, ymax_pred = int(pred_box[2]), int(pred_box[3])
    return xmin_pred, xmax_pred, ymin_pred, ymax_pred, xmin_real, xmax_real, ymin_real, ymax_real


def get_MAPE(data, ID):
    """This function calculates the MAPE.

    Args:
        data: Input dataframe.
        ID: Image Id.

    Returns:
        The return value is MAPE.

    """
    e_real = data.loc[data.im_name == ID, ['e_real']].values[0]
    e_pred = data.loc[data.im_name == ID, ['e_pred']].values[0]
    if e_real != 0:
        mape = abs(e_real - e_pred) / e_real
        return mape

    else:
        if e_pred == 0:
            return 0

        else:
            return 1


def main(real_path, pred_path):
    """This function calculates the global error: 0.7*(1-IoU) + 0.3*MAPE.

    Args:
        real_path: path to real values folder.
        pred_path: path to prediction values folder.

    Returns:
        The return value is the global error.

    """
    real_annotations = pd.read_csv(real_path + 'real.csv', sep=",")
    real_annotations.columns = ["im_name"] + [
        i + '_real' for i in real_annotations.columns if i != 'im_name'
    ]
    try:
        pred_annotations = pd.read_csv(pred_path, sep=";")
    except:
        pred_annotations = pd.read_csv(pred_path, sep=",")

    try:
        if len(pred_annotations.columns) == 6:
            pred_annotations.columns = [
                "im_name", "x_min_pred", "y_min_pred", "x_max_pred",
                "y_max_pred", "e_pred"
            ]
        elif len(pred_annotations.columns) == 7:
            pred_annotations.columns = [
                "index", "im_name", "x_min_pred", "y_min_pred", "x_max_pred",
                "y_max_pred", "e_pred"
            ]
        df = real_annotations.merge(pred_annotations, how="left", on='im_name')
        df.fillna(0, inplace=True)

        df["1_moins_IoU"] = np.nan
        df["mape"] = np.nan

        for ID in tqdm(df.im_name.tolist()):
            # print(ID)
            xmin_pred, xmax_pred, ymin_pred, ymax_pred, xmin_real, xmax_real, ymin_real, ymax_real = get_coordinates(
                df, ID)
            df.loc[df.im_name == ID, "1_moins_IoU"] = 1 - calculate_iou(
                xmin_pred, xmax_pred, ymin_pred, ymax_pred, xmin_real,
                xmax_real, ymin_real, ymax_real)
            df.loc[df.im_name == ID, "mape"] = get_MAPE(df, ID)

        global_error = 0.7 * np.nanmean(df['1_moins_IoU']) + 0.3 * np.nanmean(
            df.mape)

        return global_error

    except Exception as e:
        skipped_file = pred_path.replace("\\", "/").split("/")[-1]
        print(
            f'Skipped file: {skipped_file} is not compliant, please check it.')
        print(f'Error message: {e}')
        pass


if __name__ == '__main__':
    real_path, submissions_path = str(sys.argv[1]), str(sys.argv[2])
    #real_path, submissions_path = "./real/", "./submissions/"

    PATH = glob.glob(submissions_path + "*.csv")
    FRAME = {}
    for pred_path in PATH:
        name = pred_path.replace("\\", "/").split("/")[2].replace('.csv', "")
        print(
            f"Processing file: The {name}'s metric is currently being calculated..."
        )
        res = main(real_path, pred_path)
        FRAME[name] = res

    data = pd.DataFrame.from_dict(FRAME, orient='index').reset_index()
    data.columns = ["Team", "metric"]
    data.sort_values("metric", ascending=True).to_csv('./ranking.csv',
                                                      index=False)