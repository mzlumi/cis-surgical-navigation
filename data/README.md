# Course data

The input data and expected outputs for Programming Assignments 1 and 2 of JHU EN.601.455/655 Computer Integrated Surgery I. All coordinates are in millimetres.

The official copies are on the course schedule page (<https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:2025:fall-2025-schedule>). These files were taken from the `data` folders of a public 2022 student repository (<https://github.com/wuzijian1997/CIS1-Programming-Assignment>). Only the course-supplied files were copied, not that student's code or results.

## Sets

| Folder | Debug sets (expected output included) | Unknown sets (no answers) |
|---|---|---|
| `pa1/` | a to g, each with `-output1.txt` | h, i, j, k |
| `pa2/` | a to f, each with `-output1.txt` and `-output2.txt` | g, h, i, j |

The debug sets turn on one source of error at a time, which is what makes them useful for validation. The handout (page 13) gives the full table. In short:

| Debug set | EM distortion | EM noise | Optical tracker jiggle |
|---|---|---|---|
| a | no | no | no |
| b | no | yes | no |
| c | yes | no | no |
| d | no | no | yes |
| e | yes | no | yes |
| f, g | yes | yes | yes |

The unknown sets have all three error sources. In PA2 the registration to CT also changes from set to set.

## Files per set

| Suffix | Used in | Contents |
|---|---|---|
| `calbody` | PA1, PA2 | Calibration object geometry: optical markers on the EM base (d), LEDs on the object (a), EM markers on the object (c) |
| `calreadings` | PA1, PA2 | Per frame, the measured D, A (optical tracker) and C (EM tracker) |
| `empivot` | PA1, PA2 | Per frame, EM markers G on the EM probe during pivot calibration |
| `optpivot` | PA1, PA2 | Per frame, optical markers D on the EM base and H on the optical probe |
| `output1` | PA1 (debug), PA2 (debug) | Expected EM-pivot post position, optical-pivot post position, and per-frame expected C |
| `ct-fiducials` | PA2 | Fiducial positions b in CT coordinates |
| `em-fiducialss` | PA2 | Per fiducial, EM probe marker readings G while touching it (the double "s" is in the original file names) |
| `EM-nav` | PA2 | Per frame, EM probe readings G for navigation test points |
| `output2` | PA2 (debug) | Expected probe tip positions in CT coordinates for each `EM-nav` frame |

## Quirks to handle when parsing

- Header lines are not consistent: separators differ (`, ` or `,`), and some have a trailing comma (for example `6,pa2-debug-a-ct-fiducials.txt,`). Parse each header by splitting on commas and dropping empty fields.
- `pa2-debug-a` has no `optpivot` file. PA2 does not need optical pivot calibration, so skip it when it is missing.
- The 2020 handout lists PA1 unknown sets h and i only. This data has h to k, from a later year, but the file format is the same.
