doubleType = ["beds", "baths"]


class IgnoreAbleException(Exception):
    print("Encounter Ignoreable Exception")
    print(Exception)
    pass


def getDate(date):
    words = date.rstrip('\n').split('/')
    try:
        result = words[2] + words[0] + words[1]
        return int(result)
    except Exception as e:
        raise IgnoreAbleException(e)


def extractNumValueFromView(views):
    # City, Mountain, Park, Territorial, Water
    result = [
        1 if "city" in views else 0,
        1 if "moutain" in views else 0,
        1 if "park" in views else 0,
        1 if "territorial" in views else 0,
        1 if "water" in views else 0
    ]
    return result


def extractFloorFromAddress(address):
    words = address.split()
    result = filter(str.isdigit, words[len(words) - 1])
    try:
        return int(result)
    except Exception:
        return 0


def extractNumberOfSpacesFromParking(parking):
    words = parking.split()
    garage = False
    for i in range(len(words)):
        if words[i] == 'garage':
            garage = True
        elif i + 1 < len(words) and "space" in words[i + 1]:
            return int(words[i])
    if garage:
        return 1
    return 0


def getNumValueFromAttribute(attribute, value, past):
    value = str.lower(value)
    attribute = str.lower(attribute)
    if attribute == "type":
        if value == "condo" or value == "apartment":
            return 1
        elif value == "single family":
            return 2
        elif value == "multi family":
            return 3
        elif value == "townhouse":
            return 4
        elif value == "cooperative":
            return 5
        elif (value == "miscellaneous" or
                value == "null" or value == "multiple occupancy" or
                value == "other" or value == "mobile / manufactured"):
            raise IgnoreAbleException("Known Invalid House Type")
        else:
            raise Exception("House type unkown: {}".format(value))
    elif attribute == "latitude" or attribute == "longitude":
        return float(value)/1000000.0
    elif attribute == "address":
        if value == "":
            raise IgnoreAbleException("No address present")
        if past[len(past) - 1] == 1:
            try:
                return extractFloorFromAddress(value)
            except Exception:
                raise Exception("Attribute: [{}] = [{}] cannot be convert to float".format(attribute, value))
        else:
            return 0;
    elif attribute == "parking":
        return extractNumberOfSpacesFromParking(value)
    elif attribute == "view":
        return extractNumValueFromView(value)
    elif "lastupdatedate" in attribute:
        return getDate(value)
    else:
        try:
            if(attribute in doubleType):
                return float(value)
            else:
                return int(float(value))
        except Exception as e:
            print(e)
            raise Exception("Attribute: [{}] = [{}] cannot be convert to float".format(attribute, value))


def processRow(index, row):
    if len(index) != len(row):
        raise Exception('row elements count not equal index row element count')
    newRow = []
    price = 0
    for i in range(len(index)):
        try:
            newValue = getNumValueFromAttribute(index[i], row[i], newRow)
            if index[i] == "price":
                price = newValue
            elif type(newValue) is list:
                newRow.extend(newValue)
            else:
                newRow.append(newValue)
        except IgnoreAbleException:
            return [], price
        # print(index[i], newValue)

    return newRow, price
