#!/usr/bin/python

import math
import copy
import cv2
import numpy as np

# トーンカーブ関連の関数とクラスはそのまま残します。

# FilmicUtilsに基づくクラス定義
class CurveParamsUser:
    def __init__(self):
        self.m_toeStrength = 0.0  # 比率
        self.m_toeLength = 0.5  # 比率
        self.m_shoulderStrength = 0.0  # 白点、比率
        self.m_shoulderLength = 0.5  # Fストップ単位

        self.m_shoulderAngle = 0.0  # 比率
        self.m_gamma = 1.0


class CurveParamsDirect:
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.m_x0 = 0.25
        self.m_y0 = 0.25
        self.m_x1 = 0.75
        self.m_y1 = 0.75
        self.m_W = 1.0
        self.m_gamma = 1.0
        self.m_overshootX = 0.0
        self.m_overshootY = 0.0


class CurveSegment:
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.m_offsetX = 0.0
        self.m_offsetY = 0.0
        self.m_scaleX = 1.0
        self.m_scaleY = 1.0
        self.m_lnA = 0.0
        self.m_B = 1.0

    def Eval(self, x):
        x0 = (x - self.m_offsetX) * self.m_scaleX
        y0 = 0.0
        if (x0 > 0):
            y0 = math.exp(self.m_lnA + self.m_B * math.log(x0))

        result = y0 * self.m_scaleY + self.m_offsetY
        return result

    def EvalInv(self, y):
        y0 = (y - self.m_offsetY) / self.m_scaleY
        x0 = 0.0
        if y0 > 0:
            x0 = math.exp((math.log(y0) - self.m_lnA) / self.m_B)

        x = x0 / self.m_scaleX + self.m_offsetX
        return x


class FullCurve:
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.m_W = 1.0
        self.m_invW = 1.0
        self.m_x0 = 0.25
        self.m_y0 = 0.25
        self.m_x1 = 0.75
        self.m_y1 = 0.75

        self.m_segments = [CurveSegment() for _ in range(3)]
        self.m_invSegments = [CurveSegment() for _ in range(3)]

    def Eval(self, srcX):
        normX = srcX * self.m_invW
        if normX < self.m_x0:
            index = 0
        elif normX < self.m_x1:
            index = 1
        else:
            index = 2

        segment = self.m_segments[index]
        return segment.Eval(normX)

    def EvalInv(self, x):
        if x < self.m_y0:
            index = 0
        elif x < self.m_y1:
            index = 1
        else:
            index = 2

        segment = self.m_segments[index]
        normX = segment.EvalInv(x)
        return normX * self.m_W


def Saturate(v) :
    return max(0.0, min(1.0, v))

def powf(v, p) :
    return math.pow(v, p)

def sqrtf(v) :
    return math.sqrt(v)

def logf(v) :
    return math.log(v)

def log2f(v) :
    return math.log(v) / math.log(2)

def expf(v) :
    return math.exp(v)

def exp2f(v) :
    return 2 ** v

def MaxFloat(x, y) :
    return max(float(x), float(y))

class Vec3 :
    def __init__(self, x, y=None, z=None) :
        # "copy constructor"
        if isinstance(x, Vec3) :
            self.x = x.x
            self.y = x.y
            self.z = x.z
            return

        self.x = float(x)
        if y is None and z is None :
            self.y = float(x)
            self.z = float(x)

        else :
            self.y = float(y)
            self.z = float(z)

    def __sub__(self, other):
        if type(other) == float :
            return Vec3(self.x - other, self.y - other, self.z - other)

        raise NotImplementedError ("__sub__ undefined for Vec3 and %s" % type(other))

    def __add__(self, other):
        if type(other) == float :
            return Vec3(self.x + other, self.y + other, self.z + other)

        raise NotImplementedError( "__add__ undefined for Vec3 and %s" % type(other))

    def __radd__(self, other):
        return self + other # commutativity FTW

    def __mul__(self, other):
        if type(other) == float :
            return Vec3(self.x * other, self.y * other, self.z * other)

        raise NotImplementedError("__mul__ undefined for Vec3 and %s" % type(other))



def CreateCurve(srcParams) :
    params = copy.copy(srcParams)

    dstCurve = FullCurve()
    dstCurve.m_W = srcParams.m_W
    dstCurve.m_invW = 1.0 / srcParams.m_W

    # normalize params to 1.0 range
    params.m_W = 1.0
    params.m_x0 /= srcParams.m_W
    params.m_x1 /= srcParams.m_W
    params.m_overshootX = srcParams.m_overshootX / srcParams.m_W

    toeM = 0.0
    shoulderM = 0.0
    endpointM = 0.0

    (m, b) = AsSlopeIntercept(params.m_x0, params.m_x1, params.m_y0, params.m_y1)

    g = srcParams.m_gamma

    # base function of linear section plus gamma is
    # y = (mx+b)^g

    # which we can rewrite as
    # y = exp(g*ln(m) + g*ln(x+b/m))

    # and our evaluation function is (skipping the if parts):
    #
    #   float x0 = (x - m_offsetX)*m_scaleX
    #   y0 = expf(m_lnA + m_B*logf(x0))
    #   return y0*m_scaleY + m_offsetY

    midSegment = CurveSegment()
    midSegment.m_offsetX = -(b / m)
    midSegment.m_offsetY = 0.0
    midSegment.m_scaleX = 1.0
    midSegment.m_scaleY = 1.0
    midSegment.m_lnA = g * logf(m)
    midSegment.m_B = g

    dstCurve.m_segments[1] = midSegment

    toeM = EvalDerivativeLinearGamma(m, b, g, params.m_x0)
    shoulderM = EvalDerivativeLinearGamma(m, b, g, params.m_x1)

    # apply gamma to endpoints
    params.m_y0 = MaxFloat(1e-5, powf(params.m_y0, params.m_gamma))
    params.m_y1 = MaxFloat(1e-5, powf(params.m_y1, params.m_gamma))

    params.m_overshootY = powf(1.0 + params.m_overshootY, params.m_gamma) - 1.0

    dstCurve.m_x0 = params.m_x0
    dstCurve.m_x1 = params.m_x1
    dstCurve.m_y0 = params.m_y0
    dstCurve.m_y1 = params.m_y1

    # toe section
    toeSegment = CurveSegment()
    toeSegment.m_offsetX = 0.0
    toeSegment.m_offsetY = 0.0
    toeSegment.m_scaleX = 1.0
    toeSegment.m_scaleY = 1.0

    (toeSegment.m_lnA, toeSegment.m_B) = SolveAB(params.m_x0, params.m_y0, toeM)
    dstCurve.m_segments[0] = toeSegment

    # shoulder section

    # use the simple version that is usually too flat
    shoulderSegment = CurveSegment()

    x0 = (1.0 + params.m_overshootX) - params.m_x1
    y0 = (1.0 + params.m_overshootY) - params.m_y1

    (lnA, B) = SolveAB(x0, y0, shoulderM)

    shoulderSegment.m_offsetX = (1.0 + params.m_overshootX)
    shoulderSegment.m_offsetY = (1.0 + params.m_overshootY)

    shoulderSegment.m_scaleX = -1.0
    shoulderSegment.m_scaleY = -1.0
    shoulderSegment.m_lnA = lnA
    shoulderSegment.m_B = B

    dstCurve.m_segments[2] = shoulderSegment

    # Normalize so that we hit 1.0 at our white point. We wouldn't have do this if we
    # skipped the overshoot part.

    # evaluate shoulder at the end of the curve
    scale = dstCurve.m_segments[2].Eval(1.0)
    invScale = 1.0 / scale

    dstCurve.m_segments[0].m_offsetY *= invScale
    dstCurve.m_segments[0].m_scaleY *= invScale

    dstCurve.m_segments[1].m_offsetY *= invScale
    dstCurve.m_segments[1].m_scaleY *= invScale

    dstCurve.m_segments[2].m_offsetY *= invScale
    dstCurve.m_segments[2].m_scaleY *= invScale

    return dstCurve
def SolveAB(x0, y0, m) :
    '''
    find a function of the form:
      f(x) = e^(lnA + Bln(x))
    where
      f(0)   = 0; not really a constraint
      f(x0)  = y0
      f'(x0) = m
    '''
    B = (m*x0)/y0
    lnA = logf(y0) - B*logf(x0)
    return (lnA, B)

def AsSlopeIntercept(x0, x1, y0, y1) :
    dy = (y1-y0)
    dx = (x1-x0);

    if (dx == 0) :
        m = 1.0
    else :
        m = dy/dx

    b = y0 - x0*m

    return (m, b)


def EvalDerivativeLinearGamma(m, b, g, x) :
    return g*m*powf(m*x+b,g-1.0)

def AsSlopeIntercept(x0, x1, y0, y1) :
    dy = (y1-y0)
    dx = (x1-x0);

    if (dx == 0) :
        m = 1.0
    else :
        m = dy/dx

    b = y0 - x0*m

    return (m, b)

# トーンカーブを適用する関数
def apply_tone_curve(image, curve):
    # 入力画像を正規化してトーンカーブを適用
    height, width, channels = image.shape
    result_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            for c in range(channels):
                # 正規化された色値をトーンカーブに適用
                normalized_value = image[i, j, c] / 255.0
                result_value = curve.Eval(normalized_value)
                result_image[i, j, c] = np.clip(result_value * 255, 0, 255)

    return result_image

def CalcDirectParamsFromUser(srcParams) :
    dstParams = CurveParamsDirect()

    toeStrength = srcParams.m_toeStrength
    toeLength = srcParams.m_toeLength
    shoulderStrength = srcParams.m_shoulderStrength
    shoulderLength = srcParams.m_shoulderLength

    shoulderAngle = srcParams.m_shoulderAngle
    gamma = srcParams.m_gamma

    # This is not actually the display gamma. It's just a UI space to avoid having to
    # enter small numbers for the input.
    perceptualGamma = 2.2

    # constraints
    toeLength = powf(Saturate(toeLength),perceptualGamma)
    toeStrength = Saturate(toeStrength)
    shoulderAngle = Saturate(shoulderAngle)
    shoulderLength = max(1e-5, Saturate(shoulderLength))

    shoulderStrength = max(0.0, shoulderStrength)

    # apply base params

    # toe goes from 0 to 0.5
    x0 = toeLength * .5
    y0 = (1.0 - toeStrength) * x0; # lerp from 0 to x0

    remainingY = 1.0 - y0

    initialW = x0 + remainingY

    y1_offset = (1.0 - shoulderLength) * remainingY
    x1 = x0 + y1_offset
    y1 = y0 + y1_offset

    # filmic shoulder strength is in F stops
    extraW = exp2f(shoulderStrength)-1.0

    W = initialW + extraW

    dstParams.m_x0 = x0
    dstParams.m_y0 = y0
    dstParams.m_x1 = x1
    dstParams.m_y1 = y1
    dstParams.m_W = W

    # bake the linear to gamma space conversion
    dstParams.m_gamma = gamma

    dstParams.m_overshootX = (dstParams.m_W * 2.0) * shoulderAngle * shoulderStrength
    dstParams.m_overshootY = 0.5 * shoulderAngle * shoulderStrength

    return dstParams


# 画像をファイルから読み込み、トーンカーブを適用して保存する処理
def process_image(input_image_path, output_image_path, params_user):
    # ユーザーのパラメータからカーブパラメータを計算
    params_direct = CalcDirectParamsFromUser(params_user)

    # カーブを作成
    curve = CreateCurve(params_direct)

    # 画像の読み込み
    image = cv2.imread(input_image_path)
    if image is None:
        print("画像が読み込めませんでした。")
        return

    # トーンカーブを適用
    result_image = apply_tone_curve(image, curve)

    # 結果画像を保存
    cv2.imwrite(output_image_path, result_image)
    print(f"処理が完了しました。結果画像: {output_image_path}")


# トーンカーブのパラメータを設定
params_user = CurveParamsUser()
params_user.m_toeStrength = 0.2
params_user.m_toeLength = 0.5
params_user.m_shoulderStrength = 0.2
params_user.m_shoulderLength = 0.5
params_user.m_gamma = 2.2

# 処理したい画像のパス
input_image_path = 'l0045.png'  # 入力画像
output_image_path = 'l0045_film.png'  # 出力画像

# 画像処理を実行
process_image(input_image_path, output_image_path, params_user)
