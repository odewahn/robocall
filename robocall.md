# Rachel the Robo Caller - Analysis

At DEF CON 22, the FTC [ran a contest to help mitigate robocalls](http://www.ftc.gov/zapping-rachel). There were three rounds, the last of which was using a set of call records collected from a robocall honeypot to determine if a caller was a robocaller. See Parts I and II of the contest for details on robocaller honeypots.

The FTC gave us two sets of data, that show a phone call from one "person" to another along with the date and time. Both collections have been randomized uniquely, but the portions of area code and subscriber number were kept the same.

This Notebook details initial exploration of the data. For the follow up on predictions, check out [Modeling Rachel the Robocaller](Modeling Rachel the Robo Caller.ipynb).

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from IPython.display import Image
Image("http://www.ftc.gov/system/files/attachments/zapping-rachel/zapping-rachel-contest.jpg")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
%matplotlib inline
# Standard toolkits in pydata land
import pandas as pd
import numpy as np
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Neat little library that is a partial port of Google's libphonenumber
import phonenumbers
from phonenumbers import geocoder
# from phonenumbers import carrier
from phonenumbers import timezone
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# First pass will use a Random Forest; more on this later
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def read_FTC(dataset):
    return pd.read_csv(dataset,
                parse_dates=["DATE/TIME"],
                converters={'LIKELY ROBOCALL': lambda val: val == 'X'},
                dtype={'TO': str, 'FROM': str, 'LIKELY ROBOCALL': bool}
    )

# This assumes you have the data locally
labeled_data = read_FTC("FTC-DEFCON Data Set 1.csv")
unlabeled_data = read_FTC("FTC-DEFCON Data Set 2.csv")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data.head()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
unlabeled_data.head()
</pre>

First things to note right off the bat:

1. We could let `phonenumbers` parse the numbers out for us and cache that
2. The phone numbers are not really a numeric value and should be treated as categorical data
3. The phone number should be broken up into individual categorical features, likely:
    * Area code
    * Carrier/Subscriber
    * Not the last 4 digits though as they are randomized and need to be paired with the rest of the number to be unique
4. It is unknown whose timezone the date and time is in. They could (should?) be normalized for each calling side
5. Time zone can be extracted from the phone numbers themselves, but it says nothing about where the caller *actually* is

## Let's see how the phonenumbers library works

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Pulling a random number from the data set
fake_number = phonenumbers.parse("19188765408")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Looking back at their docs, a leading '+' and a region of None will make phonenumbers attempt to detect region, etc.
fake_number = phonenumbers.parse("+19188765408", None)
fake_number
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
fake_number.country_code
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
phonenumbers.is_valid_number(fake_number)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
geocoder.description_for_number(fake_number, "EN")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
timezone.time_zones_for_number(fake_number)
</pre>

## Picking out features for the numbers themselves

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Do they all start with a 1?
print(labeled_data["TO"].str.get(0).unique())
print(labeled_data["FROM"].str.get(0).unique())
print(unlabeled_data["TO"].str.get(0).unique())
print(unlabeled_data["FROM"].str.get(0).unique())
</pre>

Yup! This means we're using the [North American Numbering Plan](http://en.wikipedia.org/wiki/North_American_Numbering_Plan).

> The NANP is a system of numbering plan areas (NPA) using telephone numbers consisting of a three-digit area code, a three-digit central office code, and a four-digit station number. Through this plan, telephone calls can be directed to particular regions of the larger NANP public switched telephone network (PSTN), where they are further routed by the local networks. The NANP is administered by the North American Numbering Plan Administration (NANPA), a service operated by Neustar corporation. The international calling code for the NANP is 1.

Our phone number structure is then `CAAAOOONNNN` where `C` is the country code, `AAA` is the area code, `OOO` is the "central office" code (does this come from the old operator days?), and `NNNN` are the rest of the unique digits for a caller (formally called the station number). We have randomized calls though, so we'll be ignoring `NNNN` as part of any feature on their own.

Parsing the area code and central office code is trivial with Pandas semantics. However, there are a few utilities in the `phonenumbers` library that might help in a little bit though, namely:
* `geocoder.description_for_number`
* `phonenumbers.is_valid_number`
* `timezone.time_zones_for_number`

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Let's go ahead and parse all of them
# We'll parse with a leading + based on the numbers being listed with a leading country code,
# leave the second argument as None so that the phonenumbers package has to try to detect

labeled_data["TO_PARSED"] = labeled_data["TO"].apply(lambda row: phonenumbers.parse("+" + row, None))
labeled_data["FROM_PARSED"] = labeled_data["FROM"].apply(lambda row: phonenumbers.parse("+" + row, None))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["TO_VALID"] = labeled_data["TO_PARSED"].apply(lambda ph: phonenumbers.is_valid_number(ph))
labeled_data["FROM_VALID"] = labeled_data["FROM_PARSED"].apply(lambda ph: phonenumbers.is_valid_number(ph))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data.TO_VALID.unique()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data.FROM_VALID.unique()
</pre>

There are *invalid numbers in the from case*?!? What proportion of those are from the likely robocallers?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from_valid_v_robocall = pd.crosstab([labeled_data.FROM_VALID], labeled_data['LIKELY ROBOCALL'])
from_valid_v_robocall.plot(kind='bar', stacked=True, grid=False, color=["blue", "red"])
from_valid_v_robocall
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from_valid_v_robocall.div(from_valid_v_robocall.sum(1).astype(float), axis=0).plot(kind='barh', stacked=True, color=["blue", "red"])
</pre>

Come to think of it, maybe from valid is a no-good-bad-feature as the numbers are randomized but not necessarily made valid by the FTC. Hmmm... Moving on.

While we're at it, might as well make a utility function to do our cross tabulation plots against likely robocalls.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def explore_feature(df, name):
    feature_v_robocall = pd.crosstab([df[name]], df['LIKELY ROBOCALL'])
    feature_v_robocall.plot(kind='bar', stacked=True, grid=False, color=["blue", "red"])
    fvr_div = feature_v_robocall.div(feature_v_robocall.sum(1).astype(float), axis=0)
    fvr_div.plot(kind='barh', stacked=True, color=["blue", "red"])
    return feature_v_robocall
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["TO_DESCRIPTION"] = labeled_data["TO_PARSED"].apply(lambda ph: geocoder.description_for_number(ph, "EN"))
labeled_data["FROM_DESCRIPTION"] = labeled_data["FROM_PARSED"].apply(lambda ph: geocoder.description_for_number(ph, "EN"))

def get_time_zone(ph):
    tz = timezone.time_zones_for_number(ph)

labeled_data["TO_TIMEZONE"] = labeled_data["TO_PARSED"].apply(lambda ph: timezone.time_zones_for_number(ph))
labeled_data["FROM_TIMEZONE"] = labeled_data["FROM_PARSED"].apply(lambda ph: timezone.time_zones_for_number(ph))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["FROM_TIMEZONE"].unique()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
[len(x) for x in labeled_data["FROM_TIMEZONE"].unique()]
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# For ease of plotting, I'm turning that long tuple into one string and all these into strings
def get_time_zone(ph):
    # Playing fast and loose here since only one grouping had more than one timezone in one
    tz = timezone.time_zones_for_number(ph)
    if len(tz) > 1:
        tz = ("Etc/Lots",)
    return tz[0]

labeled_data["TO_TIMEZONE"] = labeled_data["TO_PARSED"].apply(lambda ph: get_time_zone(ph))
labeled_data["FROM_TIMEZONE"] = labeled_data["FROM_PARSED"].apply(lambda ph: get_time_zone(ph))
</pre>

Wow, one of those timezones is pretty much unknown.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data[labeled_data["FROM_TIMEZONE"] == "Etc/Lots"].groupby("LIKELY ROBOCALL").aggregate(sum)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, 'FROM_TIMEZONE')
</pre>

That America/Dominica one looks interesting on the last plot (percentage of likely robocall by FROM_TIMEZONE) but there is only **1** data point. That "Etc/Lots" timezone is probably interesting though.

In reality, the timezone is being pulled out from the country code + the area code. We should just use Pandas semantics on the area code.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Extract the area code using slicing since they are all regular US numbers
labeled_data["TO_AREA_CODE"] = labeled_data["TO"].str.slice(1,4)
labeled_data["FROM_AREA_CODE"] = labeled_data["FROM"].str.slice(1,4)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data.TO_AREA_CODE.describe()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data.FROM_AREA_CODE.describe()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
to_area_code_v_likely_robocall = explore_feature(labeled_data, "TO_AREA_CODE")
</pre>

Methinks there are too many area codes to visualize that. Let's look at just the subset that is potentially interesting.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
area_code_div = to_area_code_v_likely_robocall.div(to_area_code_v_likely_robocall.sum(1).astype(float), axis=0)
sample_size = to_area_code_v_likely_robocall.sum(1)
threshold = .20
min_samples = 10
threshold_true_robo = (sample_size > min_samples) & ((area_code_div[True] < threshold) | (area_code_div[True] > (1 - threshold)))

thresholded_area_robo = area_code_div[threshold_true_robo]

to_area_code_v_likely_robocall[threshold_true_robo].plot(kind='bar', stacked=True, grid=False, color=["blue", "red"])
thresholded_area_robo.plot(kind='barh', stacked=True, color=["blue", "red"])
thresholded_area_robo
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
to_area_code_v_likely_robocall[(area_code_div[True] > (1 - threshold))]
</pre>

### Do it again with office codes?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Extract the area code using slicing since they are all regular US numbers
#  labeled_data["TO_OFFICE_CODE"] = labeled_data["TO"].str.slice(4,7)
#  labeled_data["FROM_OFFICE_CODE"] = labeled_data["FROM"].str.slice(4,7)

#  Wait a second, these office codes need to be paired with their area codes. We'll have to include those.
labeled_data["TO_OFFICE_CODE"] = labeled_data["TO"].str.slice(1,7)
labeled_data["FROM_OFFICE_CODE"] = labeled_data["FROM"].str.slice(1,7)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# This is going to have the same (and worse) issue that exploring area code did.
# We'll create a thresholded utitlity function here

def explore_thresholded_feature(df, name, threshold=.20, min_samples=10):
    feature_v_robocall = pd.crosstab([df[name]], df['LIKELY ROBOCALL'])
    proportionated_feature = feature_v_robocall.div(feature_v_robocall.sum(1).astype(float), axis=0)
    sample_size = feature_v_robocall.sum(1)

    threshold_true_robo = (sample_size > min_samples) & ((proportionated_feature[True] < threshold) | (proportionated_feature[True] > (1 - threshold)))

    thresholded_feature_v_robocall = feature_v_robocall[threshold_true_robo]

    thresholded_feature_v_robocall.plot(kind='barh', stacked=True, color=["blue", "red"])
    proportionated_feature[threshold_true_robo].plot(kind='barh', stacked=True, color=["blue", "red"])

    return thresholded_feature_v_robocall



explore_thresholded_feature(labeled_data, "TO_OFFICE_CODE", threshold=.08, min_samples=25)
</pre>

Arg. Still not really easy to look at.

Did notice a few things though, namely that there are *some* area+office numbers that actually had a higher proportion of robocallers. Looks like decent numbers have no robocallers, could be one of those sections that is already populated by real people and no room to get additional numbers?

I'm tending towards using Random Forests to classify data, how well will it work when there aren't many samples for a given category?

Let's make a different version of that thresholded function now that lets you choose the direction of the threshold.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def explore_thresholded_feature(df, name, threshold=.20, min_samples=10, tend_toward_robocallers=True):
    feature_v_robocall = pd.crosstab([df[name]], df['LIKELY ROBOCALL'])
    proportionated_feature = feature_v_robocall.div(feature_v_robocall.sum(1).astype(float), axis=0)
    sample_size = feature_v_robocall.sum(1)

    # Seeking those with LOTS of robo callers
    threshold_true_robo = (proportionated_feature[True] > (1 - threshold))

    # Conditionally look at those that tend not to have robocallers
    if(not tend_toward_robocallers):
        threshold_true_robo |= proportionated_feature[True] < threshold

    # Limit by number of samples available
    threshold_true_robo &= (sample_size > min_samples)

    thresholded_feature_v_robocall = feature_v_robocall[threshold_true_robo]

    thresholded_feature_v_robocall.plot(kind='barh', stacked=True, color=["blue", "red"])
    proportionated_feature[threshold_true_robo].plot(kind='barh', stacked=True, color=["blue", "red"])

    return thresholded_feature_v_robocall

explore_thresholded_feature(labeled_data, "TO_OFFICE_CODE", threshold=.08, min_samples=25)
</pre>

That *is* an interesting collection. `786329` really stands out. We'll keep this as a categorical feature for our classifier.

## It's about time

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["DATE/TIME"]
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Extract Hour, Minute, and day of week
labeled_data["HOUR"] = labeled_data["DATE/TIME"].apply(lambda x: x.hour)
labeled_data["MINUTE"] = labeled_data["DATE/TIME"].apply(lambda x: x.minute)
labeled_data["DAYOFWEEK"] = labeled_data["DATE/TIME"].apply(lambda x: x.dayofweek)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, "HOUR")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, "MINUTE")
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["INTERVAL"] = pd.cut(labeled_data["MINUTE"], bins=range(-1,61,15), include_lowest=True)
explore_feature(labeled_data, "INTERVAL")
</pre>

Minutes probably need to be paired up with hour

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
labeled_data["TIMECHUNK"] = labeled_data["DATE/TIME"].apply(lambda x: x.hour + np.floor(4*(x.minute/60.0))/4)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, "TIMECHUNK")
</pre>

Quite similar to the hour curve, but clearly more granular.

Is there a way to track that hours wrap around? Can my classifier care that this isn't bounded at 0 and 24, that there is a modulus involved?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, "DAYOFWEEK")
</pre>

Wait! **Where is 0?** That's some strong sampling bias if 0 (Monday) isn't even included...

UPDATE: I spoke with the group that produced the data and they accidentally got rid of Monday. I do have more data I could work with in the future.

## Forest for the trees

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def total_call_volume(df):
    sizes = df.groupby("FROM").size()

    def get_size(val):
        return sizes[val]

    df["NUM_FROM_CALLS"] = df["FROM"].apply(get_size)

    sizes = df.groupby("TO").size()
    df["NUM_TO_CALLS"] = df["TO"].apply(get_size)

total_call_volume(labeled_data)
total_call_volume(unlabeled_data)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
explore_feature(labeled_data, "NUM_FROM_CALLS")
</pre>

## Summary

At this point, we've explored a few features and can build a model from them with some simple tools. Let's use those simple features to create a simple Random Forest classifier in [Modeling Rachel the Robocaller](Modeling Rachel the Robo Caller.ipynb).
