package org.example;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

abstract class PredictFunction {
    abstract float predict(Vector x, Vector w, float b);
    abstract  Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset);

    public float derivativeByB(Vector w, float b, Pair<Vector, Float>[] dataset) {
        float total = 0;
        int datasetLength = dataset.length;

        for (Pair<Vector, Float> vectorFloatPair : dataset) {
            total += (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
        }

        return total / datasetLength;
    }
}

class Vector{
    private final float[] points;
    Vector(float... points){
        this.points = points;
    }

    Vector(int size){
        points = new float[size];
    }

    float x(int i){
        return points[i];
    }

    void setX(int pos, float value){
        points[pos] = value;
    }

    int size(){
        return points.length;
    }

    float dot(Vector w){
        if(points.length != w.size()){
            return Float.NaN;
        }

        int n = points.length;
        float total = 0;
        for(int i=0;i<n;i++){
            total += points[i] * w.x(i);
        }

        return total;
    }

    void subtract(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] -= v.x(i);
        }
    }

    Vector scaleBy(float x){
        for(int i=0;i<points.length;i++){
            points[i] *= x;
        }

        return this;
    }

    float sum(){
        float total = 0;

        for (float point : points) {
            total += point;
        }

        return total;
    }
}

class PolynomialPredictor extends PredictFunction{
    @Override
    public float predict(Vector x, Vector w, float b) {
        return x.dot(w) + b;
    }

    @Override
    public Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Float> vectorFloatPair : dataset) {
                float curr = derivative.x(i);

                curr += vectorFloatPair.first.x(i) *
                        (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}

// w1*sin(x1) + w0*sin(x0) + b
class SinePredictor extends PredictFunction{
    @Override
    public float predict(Vector x, Vector w, float b) {
        return (float) (w.x(1) * Math.sin(x.x(1)) + w.x(0) * Math.sin(x.x(0)) + b);
    }

    @Override
    public Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Float> vectorFloatPair : dataset) {
                float curr = derivative.x(i);

                curr += Math.sin(vectorFloatPair.first.x(i)) * (
                        predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}

class LogisticPredictor extends PolynomialPredictor{
    @Override
    public float predict(Vector x, Vector w, float b) {
        float term = x.dot(w) + b;

        return (float) (1 / (Math.exp(-term) + 1));
    }
}

class Pair<X, Y>{
    public X first;
    public Y second;

    Pair(){

    }

    Pair(X x, Y y){
        this.first = x;
        this.second = y;
    }
}

// w = (w2, w1, w0);
// b float
// x = (x2, x1, x0)
class LinearRegression extends Regression {

    LinearRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        super(predictor, dataset);
    }

    // R-squared error
    @Override
    public float cost(){
        int n = dataset.length;
        float total = 0;

        for(Pair<Vector, Float> p : dataset){
            float term = (p.second - predictor.predict(p.first, w, b));
            total += term * term;
        }

        return total / (2 * n);
    }

    @Override
    public float predict(Vector x){
        if(x.size() != w.size()){
            return Float.NaN;
        }

        return predictor.predict(x, w, b);
    }
}

public abstract class Regression{
    protected final PredictFunction predictor;
    protected final Vector w;
    protected float b;
    protected final Pair<Vector, Float>[] dataset;
    Regression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        this.predictor = predictor;
        this.w = new Vector(dataset[0].first.size());
        this.dataset = dataset;
    }

    abstract public float cost();
    abstract protected float predict(Vector x);

    public void train(float learningRate, int iter){
        float cost;
        int iteration = 0;

        while((cost = Math.abs(cost())) > 0.0001 && iteration < iter){
            Vector v = predictor.derivativeByW(w, b, dataset).scaleBy(learningRate);

            b -= learningRate * predictor.derivativeByB(w, b, dataset);
            w.subtract(v);
            iteration++;
        }

        System.out.println(cost);
    }

    private void saveParams() throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter("w.param"));
        for(int i=0;i<w.size();i++){
            writer.write(Float.toString(w.x(i)));
            writer.newLine();
        }

        writer.close();

        writer = new BufferedWriter(new FileWriter("b.param"));
        for(int i=0;i<b;i++){
            writer.write(Float.toString(b));
            writer.newLine();
        }

        writer.close();
    }
}

class LogisticRegression extends Regression {

    LogisticRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        super(predictor, dataset);
    }

    @Override
    public float cost(){
        float total = 0;

        // -(yi * log(sigmoid) + (1 - yi)*log(1 - sigmoid))

        for(Pair<Vector, Float> point : dataset){
            float predictPercent = predictor.predict(point.first, w, b);

            total -= (point.second * Math.log(predictPercent + 0.0001))
                    + (1 - point.second) * Math.log(1 - predictPercent + 0.0001);
        }

        return total / dataset.length;
    }

    @Override
    protected float predict(Vector x) {
        return predictor.predict(x, w, b);
    }

    public boolean isPositive(Vector x){
        return predict(x) >= 0.5f;
    }
}