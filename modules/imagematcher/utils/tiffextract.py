import exifread
import piexif
from PIL import Image, TiffImagePlugin
from PIL.ExifTags import TAGS
from PIL.TiffTags import TYPES
import json

# Open the TIFF file in binary mode
file_path = "/home/hjcsteve/gigapixel/GigaStitch/stitching_pipeline/real_mesh/mappa/images/mosaico_0010.tif"
#file_path= "/home/hjcsteve/gigapixel/Stitching/GigaStitch/images/1.jpg"
OUTDIR = "exifTest/"


# Print EXIF data
# for tag, value in tags.items():
#     print(f"{tag}: {value}")

image = Image.open(file_path)

# #load exif USING PIEXIF
# piexif_exif = piexif.load(file_path)
# # for ifd in ("0th", "Exif", "GPS", "1st"):
# #     for tag in exif_dict[ifd]:
# #         print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])
# output_path = OUTDIR+"piexif.tiff"
# #modify exif
# # piexif_exif["0th"][piexif.ImageIFD.ImageWidth] = 3000
# piexiftags=piexif_exif['Exif'] 
# decoded_tags = {TAGS.get(tag, tag): value for tag, value in piexiftags.items()}

# exif_bytes = piexif.dump(piexif_exif)
# image.save(output_path, "TIFF", exif=exif_bytes)

# USING PIL
info = image.getexif()
if info is None:
    print('Sorry, image has no EXIF data.')
decoded_tags = {TAGS.get(tag, tag): value for tag, value in info.items()}
img_exif = {}
for tag, value in info.items():
    img_exif[tag] = value

output_path =  OUTDIR+"PIL_exif.tif"
image.save(output_path, "TIFF", exif=img_exif)

print("EXIF data with exifread")
with open(file_path, "rb") as file:
    tags = exifread.process_file(file)
exifread_exif = {}
for tag, value in tags.items():
    exifread_exif[tag] = value
img_exif = {}
imageFileDir = TiffImagePlugin.ImageFileDirectory_v2()
tagTypes = imageFileDir.tagtype
for tag, value in tags.items():
    # Skip tags without corresponding EXIF IDs
    if tag.startswith("EXIF "):
        # Extract tag name
        tag_name = tag.split(" ", 1)[1] if " " in tag else tag
        # if tag_name not in required_exif:
        #         continue
        # # # if value is bytes
        key = value.tag
        # if value.field_type == 5 or value.field_type == 10:
        #     img_exif[key] = float(value.printable)
        #     continue
        # if isinstance(value, bytes):
        #       img_exif[key] = value  
        #       continue
        # if len(value.values) == 0:
        #     img_exif[key] = []
        #     continue
        # if len(value.values) == 1:
        #     img_exif[key] = value.values[0]
        #     continue
        # if len(value.values) > 1 and not isinstance(value.values, str) :
        #     #convert to tuple of int
        #     for i in range(len(value.values)):
        #         value.values[i] = int(value.values[i])
        #     img_exif[key] = tuple(value.values)
        #     continue
        # #convert ratio to string
        # if len(value.values) > 1 and isinstance(value.values, str) :
        #     img_exif[key] = str(value.values)
        #     continue
        type_name = TYPES.get(value.field_type)
        values = value.values
        res = None
        match type_name:
            case 'short':
                res = values[0]
            case 'long':
                res = imageFileDir.write_long(values[0])
            case 'signed byte':
                res = imageFileDir.write_byte(values)
            case 'signed short':
                res = imageFileDir.write_short(values)
            case 'signed long':
                res = imageFileDir.write_long(values)
            case 'float':
                res = imageFileDir.write_float(values)
            case 'double':
                res = imageFileDir.write_double(values)
            case 'long':
                res = imageFileDir.write_long(values)
            case 'long8':
                res = imageFileDir.write_long8(values)
            case 'byte':
                res = imageFileDir.write_byte(values)
            case 'string':
                #res = imageFileDir.write_string(values)
                res= values
            case 'rational':
                rational = TiffImagePlugin.IFDRational(values[0].num, values[0].den)
                res = imageFileDir.write_rational(rational)
            case 'undefined':
                res = imageFileDir.write_undefined(values[0])
            case 'signed rational':
                rational = TiffImagePlugin.IFDRational(values[0].num, values[0].den)
                res = imageFileDir.write_signed_rational(rational)
            case _:
                res = "Unknown type"
        img_exif[key] =res  # Convert value to string format


output_path = OUTDIR+"exifread_exif.tiff"
image.save(output_path, "TIFF", exif=img_exif)

#verfiy exif data
test_image = Image.open(output_path)
test_info = test_image.getexif().items()
test_exif = {}
test_exif = {TAGS.get(tag, tag): value for tag, value in test_info}
exif_table = {}
items = info.items()
for tag, value in items:
    decoded = TAGS.get(tag, tag)
    print( f"{decoded} ---> {value}" )
    #skip bytes value
    if isinstance(value, bytes):
        continue
    #convert value to string even object
    value = str(value)
    exif_table[decoded] = value

pil_exif = {}
for tag, value in tags.items():
    # Skip tags without corresponding EXIF IDs
    if tag.startswith("EXIF ") or tag.startswith("Image "):
        # Extract tag name
        tag_name = tag.split(" ", 1)[1] if " " in tag else tag
        pil_exif[tag_name] = str(value)  # Convert value to string format
# Convert EXIF data to bytes
exif_bytes = piexif.dump(pil_exif)
# Save the image in TIFF format with custom EXIF data
output_path = "custom_image_piexif.tiff"
image.save(output_path, "TIFF", exif=exif_bytes)

f = open(output_path, 'rb')
tags = exifread.process_file(f)

print(f"Image saved with custom EXIF data at {output_path}")