package org.example;

//        Pair<Vector, Float>[] dataset = new Pair[]{
//                new Pair<>(new Vector(3.0f), 3.0f),
//                new Pair<>(new Vector(5.0f), 5.0f),
//                new Pair<>(new Vector(4.0f), 4.0f),
//                new Pair<>(new Vector(6.0f), 6.0f),
//                new Pair<>(new Vector(8.0f), 8.0f),
//                new Pair<>(new Vector(7.0f), 7.0f),
//        };
//
//        Regression regression = new Regression(new PolynomialPredictor(), dataset);
//        regression.train(0.01f);
//        System.out.println(regression.predict(new Vector(10.3f)));

//        Pair<Vector, Float>[] dataset = new Pair[]{
//                new Pair<>(new Vector(0, 3), 12.0f),
//                new Pair<>(new Vector(1, -2), -1.0f),
//                new Pair<>(new Vector(3, 0), 9.0f),
//                new Pair<>(new Vector(1, 1), 8.0f),
//                new Pair<>(new Vector(-1, -1), -2.0f),
//                new Pair<>(new Vector(2, 2), 13.0f),
//                new Pair<>(new Vector(-2, -2), -7.0f),
//        };
//
//        Regression regression = new Regression(new PolynomialPredictor(), dataset);
//        regression.train(0.01f);
//        System.out.println(regression.predict(new Vector(-3.0f, 4.0f)));


//        Pair<Vector, Float>[] dataset = new Pair[]{
//                new Pair<>(new Vector(0, 0, 0), 0.5f),
//                new Pair<>(new Vector(1, 0, 0), 3.5f),
//                new Pair<>(new Vector(0, 0, 1), -11.5f),
//                new Pair<>(new Vector(0, 0, 2), -23.5f),
//                new Pair<>(new Vector(1.5f, 0, 0), 5.0f),
//                new Pair<>(new Vector(1.5f, 1.0f, 0.5f), 8.0f),
//                new Pair<>(new Vector(-2.5f, 2, 4), -37.0f),
//        };
//
//        Regression regression = new Regression(new PolynomialPredictor(), dataset);
//        regression.train(0.001f);
//        System.out.println(regression.predict(new Vector(5.0f, 5.0f, 1.0f)));

//        SinePredictor predictor = new SinePredictor();
//        float pi = (float) Math.PI;
//
//        Vector w = new Vector(5.0f, -2.0f);
//        Vector[] v = new Vector[7];
//
//        v[0] = new Vector(pi, pi * 2);
//        v[1] = new Vector(pi, pi / 2);
//        v[2] = new Vector(pi, pi / 3);
//        v[3] = new Vector(2 * pi, pi / 3);
//        v[4] = new Vector(3 * pi, pi / 3);
//        v[5] = new Vector(3 * pi, pi / 4);
//        v[6] = new Vector(pi / 3, pi / 4);
//
//        Pair<Vector, Float>[] dataset = new Pair[]{
//                new Pair<>(v[0], predictor.predict(v[0], w, 5)),
//                new Pair<>(v[1], predictor.predict(v[1], w, 5)),
//                new Pair<>(v[2], predictor.predict(v[2], w, 5)),
//                new Pair<>(v[3], predictor.predict(v[3], w, 5)),
//                new Pair<>(v[4], predictor.predict(v[4], w, 5)),
//                new Pair<>(v[5], predictor.predict(v[5], w, 5)),
//                new Pair<>(v[6], predictor.predict(v[6], w, 5)),
//        };
//
//        Regression regression = new Regression(predictor, dataset);
//        regression.train(0.01f);
//        System.out.println(regression.predict(new Vector(pi / 4, pi / 2)));

import java.io.File;
import java.io.FileInputStream;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        Pair<Vector, Float>[] dataset = loadData();

        System.out.println(new LogisticRegression(new LogisticPredictor(), dataset).cost());
    }

    private static Pair<Vector, Float>[] loadData() throws IOException{
        File f = new File("test.txt");

        FileInputStream fIn = new FileInputStream(f);

        String s = new String(fIn.readAllBytes());
        String[] xAndY = s.split("\r\n\r\n");

        String[] allX = xAndY[0].split("\r\n");

        Pair<Vector, Float>[] data = new Pair[allX.length];
        int index = 0;
        for(String x : allX){
            int endOfFirst = -1, startOfSecond = -1;
            for(int i=0;i<x.length();i++){
                if(x.charAt(i) != ' '){
                    continue;
                }

                endOfFirst = i;
                for(int j=i;j<=x.length();j++){
                    if(x.charAt(j) != ' '){
                        startOfSecond = j;
                        break;
                    }
                }
                break;
            }


            data[index] = new Pair<>(new Vector(
                    Float.parseFloat(x.substring(0, endOfFirst)),
                    Float.parseFloat(x.substring(startOfSecond))), 0.0f);
            index++;
        }

        String[] allY = xAndY[1].split("\r\n");

        index = 0;
        for(String y : allY){
            String[] points = y.split(" ");

            for(String point : points){
                data[index].second = Float.parseFloat(point);
                index++;
            }
        }

        fIn.close();

        return data;
    }
}