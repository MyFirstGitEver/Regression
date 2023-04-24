package org.example;

import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;

class ExcelReader {
    private final XSSFWorkbook workbook;

    ExcelReader(String path) throws IOException {
        File file = new File(path);

        FileInputStream fIn = new FileInputStream(file);
        workbook = new XSSFWorkbook(fIn);

        fIn.close();
    }

    public Object[] getRow(int number, int sheetNum) throws Exception {
        XSSFSheet sheet = workbook.getSheetAt(sheetNum);

        int last = 0;
        try{
            last = sheet.getRow(number).getLastCellNum();
        }
        catch (Exception e){
            throw new Exception();
        }

        Object[] data = new Object[last];

        XSSFRow row = sheet.getRow(number);

        for(int i=0;i<last;i++){
            if(row.getCell(i) == null){
                data[i] = null;
                continue;
            }

            if(row.getCell(i).getCellType() == CellType.STRING){
                data[i] = row.getCell(i).getStringCellValue();
            }
            else{
                data[i] = row.getCell(i).getNumericCellValue();
            }
        }

        return data;
    }

    public int getRowCount(){
        return workbook.getSheetAt(0).getLastRowNum() + 1;
    }

    public Pair<Vector, Float>[] createLabeledDataset(
            int labelCol, int sheetNum, double negativeLabel, double customNeg, double customPos) {
        Pair<Vector, Float>[] dataset = new Pair[getRowCount() - 1];
        HashMap<String, Integer>[] hms = new HashMap[workbook.getSheetAt(sheetNum).getRow(0).getLastCellNum() + 1];

        for (int i = 0; i < dataset.length; i++) {
            Object[] data;
            try {
                data = getRow(i + 1, sheetNum);
            } catch (Exception e) {
                break;
            }

            float[]  points = new float[data.length - 1];
            float label = 0;
            int index = 0;

            for(int j=0;j<data.length;j++) {
                double numericValue;

                if(data[j] == null){
                    numericValue = 0;
                }
                else if(data[j] instanceof String){
                    if(hms[j] == null){
                        hms[j] = new HashMap<>();
                    }

                    if(hms[j].get(data[j]) == null){
                        hms[j].put((String) data[j], hms[j].size());
                    }

                    numericValue = hms[j].get(data[j]);
                }
                else{
                    numericValue = (double) data[j];
                }

                if(labelCol == j) {
                    if(numericValue == negativeLabel){
                        numericValue = customNeg;
                    }
                    else{
                        numericValue = customPos;
                    }

                    label = (float) numericValue;
                    continue;
                }

                points[index] = (float) numericValue;
                index++;
            }

            dataset[i] = new Pair<>(new Vector(points), label);
        }

        return dataset;
    }
}
