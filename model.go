package main

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	w *mat.Dense
	a Activator
	p sync.Pool
}

func NewLayer(w, b *mat.Dense, act Activator) *Layer {
	return &Layer{
		w: w,
		a: act,
		p: sync.Pool{
			New: func() interface{} { return new(tempData) },
		},
	}
}

type (
	tempData struct {
		t, h  mat.Dense
		de_dh mat.Dense
		de_dt mat.Dense
		df_t  mat.Dense
		de_dw mat.Dense
	}
	TempData struct {
		*Layer
		*tempData
	}
)

func (td TempData) Done() {
	td.t.Reset()
	td.h.Reset()
	td.de_dh.Reset()
	td.de_dt.Reset()
	td.df_t.Reset()
	td.de_dw.Reset()
	td.Layer.p.Put(td.tempData)
}

func (l *Layer) Len() int { _, cols := l.w.Dims(); return cols }

func (l *Layer) NewTempData() *TempData {
	tempData := l.p.Get().(*tempData)
	return &TempData{Layer: l, tempData: tempData}
}

func (l *Layer) ForwardWithTemp(x *mat.Dense, tmp *TempData) {
	tmp.t.Mul(x, l.w)
	l.a.F(&tmp.h, &tmp.t)
}

func (l *Layer) Backward(x *mat.Dense, this *TempData) {
	l.a.DF(&this.df_t, &this.t)
	this.de_dt.MulElem(&this.de_dh, &this.df_t)
	this.de_dw.Mul(x.T(), &this.de_dt)
}

type Inferencer struct {
	temps  []*TempData
	inputs []*mat.Dense
}

func NewInferencer(layers []*Layer) *Inferencer {
	temps := make([]*TempData, len(layers))
	for i := range layers {
		temps[i] = layers[i].NewTempData()
	}
	inputs := make([]*mat.Dense, len(layers))
	for i := 1; i < len(layers); i++ {
		inputs[i] = &temps[i-1].h
	}
	return &Inferencer{temps: temps, inputs: inputs}
}

func (i *Inferencer) Done() {
	for _, temp := range i.temps {
		temp.Done()
	}
}

func (i *Inferencer) Infer(x *mat.Dense) {
	i.inputs[0] = x
	for j := range i.temps {
		i.temps[j].ForwardWithTemp(i.inputs[j], i.temps[j])
		if j < len(i.temps)-1 {
			i.inputs[j+1] = &i.temps[j].h
		}
	}
}

func (i *Inferencer) Result(out *mat.Dense) {
	out.CloneFrom(&i.temps[len(i.temps)-1].h)
}
