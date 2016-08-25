#!/usr/bin/env python

# Copyright 2016 DIANA-HEP
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from histogrammar import *

def fill(container, datum, weight):
    if isinstance(container, Count):
        Count_fill(container, datum, weight)
    elif isinstance(container, Sum):
        Sum_fill(container, datum, weight)
    elif isinstance(container, Average):
        Average_fill(container, datum, weight)
    elif isinstance(container, Deviate):
        Deviate_fill(container, datum, weight)
    elif isinstance(container, Minimize):
        Minimize_fill(container, datum, weight)
    elif isinstance(container, Maximize):
        Maximize_fill(container, datum, weight)
    elif isinstance(container, Bag):
        Bag_fill(container, datum, weight)
    elif isinstance(container, Bin):
        Bin_fill(container, datum, weight)
    elif isinstance(container, SparselyBin):
        SparselyBin_fill(container, datum, weight)
    elif isinstance(container, CentrallyBin):
        CentrallyBin_fill(container, datum, weight)
    elif isinstance(container, IrregularlyBin):
        IrregularlyBin_fill(container, datum, weight)
    elif isinstance(container, Categorize):
        Categorize_fill(container, datum, weight)
    elif isinstance(container, Fraction):
        Fraction_fill(container, datum, weight)
    elif isinstance(container, Stack):
        Stack_fill(container, datum, weight)
    elif isinstance(container, Select):
        Select_fill(container, datum, weight)
    elif isinstance(container, Label):
        Label_fill(container, datum, weight)
    elif isinstance(container, UntypedLabel):
        UntypedLabel_fill(container, datum, weight)
    elif isinstance(container, Index):
        Index_fill(container, datum, weight)
    elif isinstance(container, Branch):
        Branch_fill(container, datum, weight)
    else:
        raise Exception

def combine(one, two):
    if isinstance(one, Count) and isinstance(two, Count):
        return Count_combine(one, two)
    elif isinstance(one, Sum) and isinstance(two, Sum):
        return Sum_combine(one, two)
    elif isinstance(one, Average) and isinstance(two, Average):
        return Average_combine(one, two)
    elif isinstance(one, Deviate) and isinstance(two, Deviate):
        return Deviate_combine(one, two)
    elif isinstance(one, Minimize) and isinstance(two, Minimize):
        return Minimize_combine(one, two)
    elif isinstance(one, Maximize) and isinstance(two, Maximize):
        return Maximize_combine(one, two)
    elif isinstance(one, Bag) and isinstance(two, Bag):
        return Bag_combine(one, two)
    elif isinstance(one, Bin) and isinstance(two, Bin):
        return Bin_combine(one, two)
    elif isinstance(one, SparselyBin) and isinstance(two, SparselyBin):
        return SparselyBin_combine(one, two)
    elif isinstance(one, CentrallyBin) and isinstance(two, CentrallyBin):
        return CentrallyBin_combine(one, two)
    elif isinstance(one, IrregularlyBin) and isinstance(two, IrregularlyBin):
        return IrregularlyBin_combine(one, two)
    elif isinstance(one, Categorize) and isinstance(two, Categorize):
        return Categorize_combine(one, two)
    elif isinstance(one, Fraction) and isinstance(two, Fraction):
        return Fraction_combine(one, two)
    elif isinstance(one, Stack) and isinstance(two, Stack):
        return Stack_combine(one, two)
    elif isinstance(one, Select) and isinstance(two, Select):
        return Select_combine(one, two)
    elif isinstance(one, Label) and isinstance(two, Label):
        return Label_combine(one, two)
    elif isinstance(one, UntypedLabel) and isinstance(two, UntypedLabel):
        return UntypedLabel_combine(one, two)
    elif isinstance(one, Index) and isinstance(two, Index):
        return Index_combine(one, two)
    elif isinstance(one, Branch) and isinstance(two, Branch):
        return Branch_combine(one, two)
    else:
        raise Exception

# Literally copied from histogrammar-docs/specification/index.md
# keep up to date!

def Count_fill(counting, datum, weight):
    if weight > 0.0:
        counting.entries += counting.transform(weight)

def Count_combine(one, two):
    out = Count.ed(one.entries + two.entries)
    out.transform = one.transform
    return out

def Sum_fill(summing, datum, weight):
    if weight > 0.0:
        q = summing.quantity(datum)
        summing.entries += weight
        summing.sum += q * weight

def Sum_combine(one, two):
    out = Sum.ed(one.entries + two.entries, one.sum + two.sum)
    out.quantity = one.quantity
    return out

def Average_fill(averaging, datum, weight):
    if weight > 0.0:
        q = averaging.quantity(datum)
        if averaging.entries == 0.0:
            averaging.mean = 0.0               # make it not NaN (has no weight in total)
        averaging.entries += weight

        if math.isnan(averaging.mean) or math.isnan(q):
            averaging.mean = float("nan")

        elif math.isinf(averaging.mean) or math.isinf(q):
            if math.isinf(averaging.mean) and math.isinf(q) and averaging.mean * q < 0.0:
                averaging.mean = float("nan")  # opposite-sign infinities is bad
            elif math.isinf(q):
                averaging.mean = q             # mean becomes infinite with sign of q
            else:
                pass                           # mean is already infinite
            if math.isinf(averaging.entries) or math.isnan(averaging.entries):
                averaging.mean = float("nan")  # non-finite denominator is bad

        else:                                  # handle finite case
            delta = q - averaging.mean
            shift = delta * weight / averaging.entries
            averaging.mean += shift

def Average_combine(one, two):
    entries = one.entries + two.entries
    if one.entries == 0.0:
        mean = two.mean
    elif two.entries == 0.0:
        mean = one.mean
    else:
        mean = (one.entries*one.mean + two.entries*two.mean)/entries
    out = Average.ed(entries, mean)
    out.quantity = one.quantity
    return out

def Deviate_fill(deviating, datum, weight):
    if weight > 0.0:
        q = deviating.quantity(datum)
        if deviating.entries == 0.0:
            deviating.mean = 0.0              # make it not NaN (has no weight in total)
            deviating.variance = 0.0
        varianceTimesEntries = deviating.variance * deviating.entries
        deviating.entries += weight

        if math.isnan(deviating.mean) or math.isnan(q):
            deviating.mean = float("nan")
            varianceTimesEntries = float("nan")

        elif math.isinf(deviating.mean) or math.isinf(q):
            if math.isinf(deviating.mean) and math.isinf(q) and deviating.mean * q < 0.0:
                deviating.mean = float("nan") # opposite-sign infinities is bad
            elif math.isinf(q):
                deviating.mean = q            # mean becomes infinite with sign of q
            else:
                pass                          # mean and variance are already infinite
            if math.isinf(deviating.entries) or math.isnan(deviating.entries):
                deviating.mean = float("nan") # non-finite denominator is bad

            # any infinite value makes the variance NaN
            varianceTimesEntries = float("nan")

        else:                                 # handle finite case
            delta = q - deviating.mean
            shift = delta * weight / deviating.entries
            deviating.mean += shift
            varianceTimesEntries += weight * delta * (q - deviating.mean)

        deviating.variance = varianceTimesEntries / deviating.entries

def Deviate_combine(one, two):
    entries = one.entries + two.entries
    if one.entries == 0.0:
        mean = two.mean
        varianceTimesEntries = two.variance * two.entries
    elif two.entries == 0.0:
        mean = one.mean
        varianceTimesEntries = one.variance * one.entries
    else:
        mean = (one.entries*one.mean + two.entries*two.mean) / entries
        varianceTimesEntries = one.entries*one.variance + two.entries*two.variance \
                               + one.entries*one.mean*one.mean + two.entries*two.mean*two.mean \
                               - 2.0*mean*(one.entries*one.mean + two.entries*two.mean) \
                               + entries*mean*mean

    if entries == 0.0:
        variance = varianceTimesEntries
    else:
        variance = varianceTimesEntries / entries

    out = Deviate.ed(entries, mean, variance)
    out.quantity = one.quantity
    return out

def Minimize_fill(minimizing, datum, weight):
    if weight > 0.0:
        q = minimizing.quantity(datum)
        minimizing.entries += weight
        if math.isnan(minimizing.min) or q < minimizing.min:
            minimizing.min = q

def Minimize_combine(one, two):
    entries = one.entries + two.entries
    if math.isnan(one.min):
        min = two.min
    elif math.isnan(two.min):
        min = one.min
    elif one.min < two.min:
        min = one.min
    else:
        min = two.min
    out = Minimize.ed(entries, min)
    out.quantity = one.quantity
    return out

def Maximize_fill(maximizing, datum, weight):
    if weight > 0.0:
        q = maximizing.quantity(datum)
        maximizing.entries += weight
        if math.isnan(maximizing.max) or q > maximizing.max:
            maximizing.max = q

def Maximize_combine(one, two):
    entries = one.entries + two.entries
    if math.isnan(one.max):
        max = two.max
    elif math.isnan(two.max):
        max = one.max
    elif one.max > two.max:
        max = one.max
    else:
        max = two.max
    out = Maximize.ed(entries, max)
    out.quantity = one.quantity
    return out

def Bag_fill(bagging, datum, weight):
    if weight > 0.0:
        q = bagging.quantity(datum)
        if bagging.range == "N":
            if math.isnan(q):   # something to avoid NaN != NaN
                q = "nan"       # (handling is more complex in type-safe languages)
        elif bagging.range[0] == "N":
            q = tuple("nan" if math.isnan(qi) else qi for qi in q)
        bagging.entries += weight
        if q in bagging.values:
            bagging.values[q] += weight
        else:
            bagging.values[q] = weight

def Bag_combine(one, two):
    if one.range != two.range:
        raise Exception
    entries = one.entries + two.entries
    values = {}
    for v in set(one.values.keys()).union(set(two.values.keys())):
        if v in one.values and v in two.values:
            values[v] = one.values[v] + two.values[v]
        elif v in one.values:
            values[v] = one.values[v]
        elif v in two.values:
            values[v] = two.values[v]
    return Bag.ed(entries, values, one.range)

def Bin_fill(binning, datum, weight):
    if weight > 0.0:
        q = binning.quantity(datum)
        if math.isnan(q):
            fill(binning.nanflow, datum, weight)
        elif q < binning.low:
            fill(binning.underflow, datum, weight)
        elif q >= binning.high:
            fill(binning.overflow, datum, weight)
        else:
            bin = int(math.floor(binning.num * \
                (q - binning.low) / (binning.high - binning.low)))
            fill(binning.values[bin], datum, weight)
        binning.entries += weight

def Bin_combine(one, two):
    if one.num != two.num or one.low != two.low or one.high != two.high:
        raise Exception
    entries = one.entries + two.entries
    values = [combine(x, y) for x, y in zip(one.values, two.values)]
    underflow = combine(one.underflow, two.underflow)
    overflow = combine(one.overflow, two.overflow)
    nanflow = combine(one.nanflow, two.nanflow)
    return Bin.ed(one.low, one.high, entries, values, underflow, overflow, nanflow)

def SparselyBin_fill(sparselybinning, datum, weight):
    if weight > 0.0:
        q = sparselybinning.quantity(datum)
        if math.isnan(q):
            fill(sparselybinning.nanflow, datum, weight)
        else:
            softbin = (q - sparselybinning.origin) / sparselybinning.binWidth

            if softbin <= -(2**63 - 1):
                bin = -(2**63 - 1)
            elif softbin >= (2**63 - 1):
                bin = (2**63 - 1)
            else:
                bin = int(math.floor(softbin))

            if bin not in sparselybinning.bins:
                sparselybinning.bins[bin] = sparselybinning.value.copy()

            fill(sparselybinning.bins[bin], datum, weight)

        sparselybinning.entries += weight

def SparselyBin_combine(one, two):
    if one.binWidth != two.binWidth or one.origin != two.origin or one.contentType != two.contentType:
        raise Exception
    entries = one.entries + two.entries
    contentType = one.contentType
    bins = {}
    for key in set(one.bins.keys()).union(set(two.bins.keys())):
        if key in one.bins and key in two.bins:
            bins[key] = combine(one.bins[key], two.bins[key])
        elif key in one.bins:
            bins[key] = one.bins[key].copy()
        elif key in two.bins:
            bins[key] = two.bins[key].copy()
    nanflow = combine(one.nanflow, two.nanflow)
    return SparselyBin.ed(one.binWidth, entries, contentType, \
                          bins, nanflow, one.origin)

def CentrallyBin_fill(centrallybinning, datum, weight):
    if weight > 0.0:
        q = centrallybinning.quantity(datum)
        if math.isnan(q):
            fill(centrallybinning.nanflow, datum, weight)
        else:
            for index in range(len(centrallybinning.bins)):
                if index == len(centrallybinning.bins) - 1:
                    break
                thisCenter = centrallybinning.bins[index][0]
                nextCenter = centrallybinning.bins[index + 1][0]
                if q < (thisCenter + nextCenter)/2.0:
                    break
            fill(centrallybinning.bins[index][1], datum, weight)
        centrallybinning.entries += weight

def CentrallyBin_combine(one, two):
    if set(one.centers) != set(two.centers):
        raise Exception
    entries = one.entries + two.entries
    bins = []
    for c1, v1 in one.bins:
        v2 = [v for c2, v in two.bins if c1 == c2][0]
        bins.append((c1, combine(v1, v2)))

    nanflow = combine(one.nanflow, two.nanflow)
    return CentrallyBin.ed(entries, bins, nanflow)

def IrregularlyBin_fill(irregularlybinning, datum, weight):
    if weight > 0.0:
        q = irregularlybinning.quantity(datum)
        if math.isnan(q):
            fill(irregularlybinning.nanflow, datum, weight)
        else:
            lowEdges = irregularlybinning.bins
            highEdges = list(irregularlybinning.bins[1:]) + [(float("nan"), None)]
            for (low, sub), (high, _) in zip(lowEdges, highEdges):
                if q >= low and not q >= high:    # include high endpoint only for the last bin
                    fill(sub, datum, weight)
                    break
        irregularlybinning.entries += weight

def IrregularlyBin_combine(one, two):
    if one.thresholds != two.thresholds:
        raise Exception
    entries = one.entries + two.entries
    bins = [(c, combine(v1, v2)) for (c, v1), (_, v2) in zip(one.bins, two.bins)]
    nanflow = combine(one.nanflow, two.nanflow)
    return IrregularlyBin.ed(entries, bins, nanflow)

def Categorize_fill(categorizing, datum, weight):
    if weight > 0.0:
        q = categorizing.quantity(datum)
        if q not in categorizing.pairs:
            categorizing.pairs[q] = categorizing.value.copy()
        fill(categorizing.pairs[q], datum, weight)
        categorizing.entries += weight

def Categorize_combine(one, two):
    if one.contentType != two.contentType:
        raise Exception
    entries = one.entries + two.entries
    contentType = one.contentType
    pairs = {}
    for key in set(one.pairs.keys()).union(set(two.pairs.keys())):
        if key in one.pairs and key in two.pairs:
            pairs[key] = combine(one.pairs[key], two.pairs[key])
        elif key in one.pairs:
            pairs[key] = one.pairs[key].copy()
        elif key in two.pairs:
            pairs[key] = two.pairs[key].copy()
    return Categorize.ed(entries, contentType, pairs)

def Fraction_fill(fractioning, datum, weight):
    if weight > 0.0:
        w = weight * fractioning.quantity(datum)
        fill(fractioning.denominator, datum, weight)
        if w > 0.0:
            fill(fractioning.numerator, datum, w)
        fractioning.entries += weight

def Fraction_combine(one, two):
    entries = one.entries + two.entries
    numerator = combine(one.numerator, two.numerator)
    denominator = combine(one.denominator, two.denominator)
    return Fraction.ed(entries, numerator, denominator)

def Stack_fill(stacking, datum, weight):
    if weight > 0.0:
        q = stacking.quantity(datum)
        if math.isnan(q):
            fill(stacking.nanflow, datum, weight)
        else:
            for threshold, sub in stacking.bins:
                if q >= threshold:
                    fill(sub, datum, weight)
        stacking.entries += weight

def Stack_combine(one, two):
    if [c for c, v in one.bins] != [c for c, v in two.bins]:
        raise Exception
    entries = one.entries + two.entries
    bins = []
    for (c, v1), (_, v2) in zip(one.bins, two.bins):
        bins.append((c, combine(v1, v2)))
    nanflow = combine(one.nanflow, two.nanflow)
    return Stack.ed(entries, bins, nanflow)

def Select_fill(selecting, datum, weight):
    if weight > 0.0:
        w = weight * selecting.quantity(datum)
        if w > 0.0:
            fill(selecting.cut, datum, w)
        selecting.entries += weight

def Select_combine(one, two):
    entries = one.entries + two.entries
    cut = combine(one.cut, two.cut)
    return Select.ed(entries, cut)

def Label_fill(labeling, datum, weight):
    if weight > 0.0:
        for _, v in labeling.pairs.items():
            fill(v, datum, weight)
        labeling.entries += weight

def Label_combine(one, two):
    if set(one.pairs.keys()) != set(two.pairs.keys()):
        raise Exception
    entries = one.entries + two.entries
    pairs = {}
    for l, v1 in one.pairs.items():
        v2 = two.pairs[l]
        pairs[l] = combine(v1, v2)
    return Label.ed(entries, pairs)

def UntypedLabel_fill(untypedlabeling, datum, weight):
    if weight > 0.0:
        for _, v in untypedlabeling.pairs.items():
            fill(v, datum, weight)
        untypedlabeling.entries += weight

def UntypedLabel_combine(one, two):
    if set(one.pairs.keys()) != set(two.pairs.keys()):
        raise Exception
    entries = one.entries + two.entries
    pairs = {}
    for l, v1 in one.pairs.items():
        v2 = two.pairs[l]
        pairs[l] = combine(v1, v2)
    return UntypedLabel.ed(entries, pairs)

def Index_fill(indexing, datum, weight):
    if weight > 0.0:
        for v in indexing.values:
            fill(v, datum, weight)
        indexing.entries += weight

def Index_combine(one, two):
    if len(one.values) != len(two.values):
        raise Exception
    entries = one.entries + two.entries
    values = []
    for v1, v2 in zip(one.values, two.values):
        values.append(combine(v1, v2))
    return Index.ed(entries, values)

def Branch_fill(branching, datum, weight):
    if weight > 0.0:
        for v in branching.values:
            fill(v, datum, weight)
        branching.entries += weight

def Branch_combine(one, two):
    if len(one.values) != len(two.values):
        raise Exception
    entries = one.entries + two.entries
    values = []
    for v1, v2 in zip(one.values, two.values):
        values.append(combine(v1, v2))
    return Branch.ed(entries, values)
