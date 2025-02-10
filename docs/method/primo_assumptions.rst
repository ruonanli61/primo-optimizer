PRIMO Assumptions
-----------------
The underlying assumptions in the PRIMO framework are summarized below:
- Absolute values of elevation delta are used for efficiency calculations. Therefore, the user-specified elevation delta values are
  always converted to non-negative numbers in the `WellData` object. This implies that incline and decline are equally efficient
  as long as the magnitude of the elevation difference is the same.
