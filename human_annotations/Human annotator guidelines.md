# Human Annotator Guidelines

### **Primary Objective**

The core task is to determine if a given **`Reference`** appears within an image generated from a corresponding **`Source`** text.

### **Annotation Columns**

For each item, you will work with two main columns:

**`Reference_in_Source`**: This column answers the main question of whether the reference is present.


### **Guideline 1: The Main Task - Identifying the Reference**

Your primary goal is to answer the question: **Does the `Reference` appear within the `Source`?**

* **If the `Reference` appears in the `Source`**: The annotation for the `Reference_in_Source` column should be **1**.
* **If the `Reference` does NOT appear in the `Source`**: The annotation for the `Reference_in_Source` column should be **0**.

* **Rule**: If it's clear a change has occurred, but the resulting text is no longer a direct reference, the annotation for the `Reference_in_Source` column must be **0**.
