package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)

type Trainer struct {
	LearnW float64
}

func NewTrainer(learnW float64) *Trainer {
	return &Trainer{
		LearnW: learnW,
	}
}

func (t *Trainer) learnW() float64 {
	if t.LearnW > 0 {
		return t.LearnW
	}
	panic("learning rate for weights must be set")
}

func (t *Trainer) Train(layers []*Layer, data Dataset, window, epochStart, epochs int, renderLoss func(loss plotter.XY)) {
	in := NewInferencer(layers)
	defer in.Done()
	x, y := data.trainingData(window)
	for nEpoch := epochStart; nEpoch < epochStart+epochs; nEpoch++ {
		// TODO: Shuffle here.
		in.Infer(x)
		// Backward pass.
		// Handling the last layer separately, because it uses the labels.
		// This implementations assumes that the last layer is a softmax layer
		// with cross-entropy loss.
		last, prev := in.temps[len(in.temps)-1], in.temps[len(in.temps)-2]
		last.de_dt.Sub(&last.h, y)

		renderLoss(plotter.XY{
			X: float64(nEpoch),
			Y: crossEntropy(&last.h, y),
		})

		last.de_dw.Mul(prev.h.T(), &last.de_dt)
		for i := len(in.temps) - 2; i >= 0; i-- {
			this := in.temps[i]
			this.de_dh.Mul(&last.de_dt, last.w.T())
			layers[i].Backward(in.inputs[i], this)
			last = this
		}
		// Update weights and biases.
		for i := range layers {
			l := layers[i]
			in.temps[i].de_dw.Scale(t.learnW(), &in.temps[i].de_dw)
			l.w.Sub(l.w, &in.temps[i].de_dw)
		}
	}
}

func crossEntropy(z, y *mat.Dense) (res float64) {
	r, c := z.Dims()
	var loss float64
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			zi := z.At(i, j)
			yi := y.At(i, j)
			loss += yi*math.Log(zi) + (1-yi)*math.Log(1-zi)
		}
	}
	return -loss / float64(r)
}

func indexOfMax(x *mat.VecDense) (idx int) {
	for i := range x.Len() {
		if x.AtVec(i) > x.AtVec(idx) {
			idx = i
		}
	}
	return idx
}
