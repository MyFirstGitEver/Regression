package org.example;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

abstract class PredictFunction {
    abstract  BigDecimal predict(Vector x, Vector w, float b);
    abstract  Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset);

    public float derivativeByB(Vector w, float b, Pair<Vector, Float>[] dataset) {
        float total = 0;
        int datasetLength = dataset.length;

        for (Pair<Vector, Float> vectorFloatPair : dataset) {
            total += (predict(vectorFloatPair.first, w, b).floatValue() - vectorFloatPair.second);
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
    public BigDecimal predict(Vector x, Vector w, float b) {
        return BigDecimal.valueOf(x.dot(w) + b);
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
                        (predict(vectorFloatPair.first, w, b).floatValue() - vectorFloatPair.second);
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
    public BigDecimal predict(Vector x, Vector w, float b) {
        return BigDecimal.valueOf(
                (float) (w.x(1) * Math.sin(x.x(1)) + w.x(0) * Math.sin(x.x(0)) + b));
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
                        predict(vectorFloatPair.first, w, b).floatValue() - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}

class LogisticPredictor extends PolynomialPredictor{
    @Override
    public BigDecimal predict(Vector x, Vector w, float b) {
        float term = x.dot(w) + b;

        BigDecimal one = new BigDecimal("1");
        BigDecimal result = one.divide(BigDecimalMath.exp(
                BigDecimal.valueOf(-term).setScale(10, RoundingMode.HALF_EVEN)).add(one),
                new MathContext(10, RoundingMode.HALF_EVEN));

        return result.setScale(10, RoundingMode.HALF_EVEN);
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
            float term = (p.second - predictor.predict(p.first, w, b).floatValue());
            total += term * term;
        }

        return total / (2 * n);
    }

    @Override
    public BigDecimal predict(Vector x){
        if(x.size() != w.size()){
            return new BigDecimal("NaN");
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
    abstract protected BigDecimal predict(Vector x);

    public void train(float learningRate){
        float cost;
        int iteration = 0;

        while((cost = Math.abs(cost())) > 0.0001 && iteration < 3000){
            Vector v = predictor.derivativeByW(w, b, dataset).scaleBy(learningRate);

            b -= learningRate * predictor.derivativeByB(w, b, dataset);
            w.subtract(v);
            iteration++;
        }

        System.out.println(cost);
    }
}

class LogisticRegression extends Regression{

    LogisticRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        super(predictor, dataset);
    }

    @Override
    public float cost(){
        float total = 0;

        // -(yi * log(sigmoid) + (1 - yi)*log(1 - sigmoid))

        BigDecimal one = new BigDecimal("1");

        for(Pair<Vector, Float> point : dataset){
            BigDecimal predictPercent = predictor.predict(point.first, w, b);

            total -= (point.second * BigDecimalMath.log(predictPercent).floatValue())
                    + (1 - point.second) * BigDecimalMath.log(one.subtract(predictPercent)).floatValue();
        }

        return total / dataset.length;
    }

    @Override
    protected BigDecimal predict(Vector x) {
        return predictor.predict(x, w, b);
    }

    public boolean isPositive(Vector x){
        return predict(x).floatValue() >= 0.5f;
    }
}