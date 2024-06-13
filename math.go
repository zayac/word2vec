package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Activator interface {
	F(*mat.Dense, *mat.Dense)
	DF(*mat.Dense, *mat.Dense)
}

type SigmoidActivator struct{}

func (SigmoidActivator) F(dst, src *mat.Dense)  { apply(dst, src, Sigmoid) }
func (SigmoidActivator) DF(dst, src *mat.Dense) { apply(dst, src, DSigmoid) }

type LReLUActivator struct{}

func (LReLUActivator) F(dst, src *mat.Dense)  { apply(dst, src, LReLU) }
func (LReLUActivator) DF(dst, src *mat.Dense) { apply(dst, src, DLReLU) }

type SoftmaxActivator struct{}

func (SoftmaxActivator) F(dst, src *mat.Dense)  { applySoftmax(dst, src) }
func (SoftmaxActivator) DF(dst, src *mat.Dense) { panic("not implemented") }

type LinearActivator struct{}

func (LinearActivator) F(dst, src *mat.Dense)  { apply(dst, src, Linear) }
func (LinearActivator) DF(dst, src *mat.Dense) { apply(dst, src, DLinear) }

func apply(out, x *mat.Dense, f func(float64) float64) {
	out.Apply(func(_, _ int, v float64) float64 { return f(v) }, x)
}

func applySoftmax(z, x *mat.Dense) {
	z.Apply(func(_, _ int, v float64) float64 { return math.Exp(v) }, x)
	for i := range z.RawMatrix().Rows {
		row := z.RowView(i).(*mat.VecDense)
		row.ScaleVec(1/mat.Sum(row), row)
	}
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func LReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.2 * x
}

func DLReLU(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.2
}

func Linear(x float64) float64 {
	return x
}

func DLinear(x float64) float64 {
	return 1
}
