#include "gen_exported.h"

namespace gen_exported {

/****************************************************************************************
Copyright (c) 2023 Cycling '74

The code that Max generates automatically and that end users are capable of
exporting and using, and any associated documentation files (the “Software”)
is a work of authorship for which Cycling '74 is the author and owner for
copyright purposes.

This Software is dual-licensed either under the terms of the Cycling '74
License for Max-Generated Code for Export, or alternatively under the terms
of the General Public License (GPL) Version 3. You may use the Software
according to either of these licenses as it is most appropriate for your
project on a case-by-case basis (proprietary or not).

A) Cycling '74 License for Max-Generated Code for Export

A license is hereby granted, free of charge, to any person obtaining a copy
of the Software (“Licensee”) to use, copy, modify, merge, publish, and
distribute copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The Software is licensed to Licensee for all uses that do not include the sale,
sublicensing, or commercial distribution of software that incorporates this
source code. This means that the Licensee is free to use this software for
educational, research, and prototyping purposes, to create musical or other
creative works with software that incorporates this source code, or any other
use that does not constitute selling software that makes use of this source
code. Commercial distribution also includes the packaging of free software with
other paid software, hardware, or software-provided commercial services.

For entities with UNDER 200k USD in annual revenue or funding, a license is hereby
granted, free of charge, for the sale, sublicensing, or commercial distribution
of software that incorporates this source code, for as long as the entity's
annual revenue remains below 200k USD annual revenue or funding.

For entities with OVER 200k USD in annual revenue or funding interested in the
sale, sublicensing, or commercial distribution of software that incorporates
this source code, please send inquiries to licensing (at) cycling74.com.

The above copyright notice and this license shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Please see
https://support.cycling74.com/hc/en-us/articles/360050779193-Gen-Code-Export-Licensing-FAQ
for additional information

B) General Public License Version 3 (GPLv3)
Details of the GPLv3 license can be found at: https://www.gnu.org/licenses/gpl-3.0.html
****************************************************************************************/

// global noise generator
Noise noise;
static const int GENLIB_LOOPCOUNT_BAIL = 100000;


// The State struct contains all the state and procedures for the gendsp kernel
typedef struct State {
	CommonState __commonstate;
	Change __m_change_14;
	Change __m_change_13;
	Data m_storage_3;
	int __exception;
	int vectorsize;
	PlusEquals __m_pluseq_16;
	PlusEquals __m_pluseq_19;
	t_sample __m_latch_15;
	t_sample __m_carry_11;
	t_sample __m_carry_6;
	t_sample __m_count_4;
	t_sample samplerate;
	t_sample __m_count_9;
	t_sample m_window_1;
	t_sample m_shift_2;
	t_sample __m_latch_20;
	// re-initialize all member variables;
	inline void reset(t_param __sr, int __vs) {
		__exception = 0;
		vectorsize = __vs;
		samplerate = __sr;
		m_window_1 = ((int)4000);
		m_shift_2 = ((int)4);
		m_storage_3.reset("storage", ((int)100000), ((int)1));
		__m_count_4 = 0;
		__m_carry_6 = 0;
		__m_count_9 = 0;
		__m_carry_11 = 0;
		__m_change_13.reset(0);
		__m_change_14.reset(0);
		__m_latch_15 = 0;
		__m_pluseq_16.reset(0);
		__m_pluseq_19.reset(0);
		__m_latch_20 = 0;
		genlib_reset_complete(this);
		
	};
	// the signal processing routine;
	inline int perform(t_sample ** __ins, t_sample ** __outs, int __n) {
		vectorsize = __n;
		const t_sample * __in1 = __ins[0];
		t_sample * __out1 = __outs[0];
		if (__exception) {
			return __exception;
			
		} else if (( (__in1 == 0) || (__out1 == 0) )) {
			__exception = GENLIB_ERR_NULL_BUFFER;
			return __exception;
			
		};
		int storage_dim = m_storage_3.dim;
		int storage_channels = m_storage_3.channels;
		int dim_149 = storage_dim;
		t_sample sub_125 = (m_shift_2 - ((int)1));
		t_sample mul_127 = (m_window_1 * sub_125);
		t_sample rdiv_140 = safediv(((int)2), m_window_1);
		// the main sample loop;
		while ((__n--)) {
			const t_sample in1 = (*(__in1++));
			__m_count_4 = (((int)0) ? 0 : (fixdenorm(__m_count_4 + ((int)1))));
			int carry_5 = 0;
			if ((((int)0) != 0)) {
				__m_count_4 = 0;
				__m_carry_6 = 0;
				
			} else if (((dim_149 > 0) && (__m_count_4 >= dim_149))) {
				int wraps_7 = (__m_count_4 / dim_149);
				__m_carry_6 = (__m_carry_6 + wraps_7);
				__m_count_4 = (__m_count_4 - (wraps_7 * dim_149));
				carry_5 = 1;
				
			};
			int counter_122 = __m_count_4;
			int counter_123 = carry_5;
			int counter_124 = __m_carry_6;
			bool index_ignore_8 = ((counter_122 >= storage_dim) || (counter_122 < 0));
			if ((!index_ignore_8)) {
				m_storage_3.write(in1, counter_122, 0);
				
			};
			t_sample sub_126 = (counter_122 - mul_127);
			__m_count_9 = (((int)0) ? 0 : (fixdenorm(__m_count_9 + ((int)1))));
			int carry_10 = 0;
			if ((((int)0) != 0)) {
				__m_count_9 = 0;
				__m_carry_11 = 0;
				
			} else if (((m_window_1 > 0) && (__m_count_9 >= m_window_1))) {
				int wraps_12 = (__m_count_9 / m_window_1);
				__m_carry_11 = (__m_carry_11 + wraps_12);
				__m_count_9 = (__m_count_9 - (wraps_12 * m_window_1));
				carry_10 = 1;
				
			};
			int counter_118 = __m_count_9;
			int counter_119 = carry_10;
			int counter_120 = __m_carry_11;
			t_sample mul_146 = (counter_118 * rdiv_140);
			t_sample mul_141 = (mul_146 * ((t_sample)3.1415926535898));
			t_sample cos_144 = cos(mul_141);
			t_sample add_143 = (cos_144 + ((int)1));
			t_sample mul_142 = (add_143 * ((t_sample)0.5));
			int change_134 = __m_change_13(cos_144);
			int change_133 = __m_change_14(change_134);
			int eq_132 = (change_133 == (-1));
			__m_latch_15 = ((eq_132 != 0) ? sub_126 : __m_latch_15);
			t_sample latch_139 = __m_latch_15;
			t_sample plusequals_130 = __m_pluseq_16.post(m_shift_2, eq_132, 0);
			int index_trunc_17 = fixnan(floor((plusequals_130 + latch_139)));
			int index_wrap_18 = ((index_trunc_17 < 0) ? ((storage_dim - 1) + ((index_trunc_17 + 1) % storage_dim)) : (index_trunc_17 % storage_dim));
			// samples storage channel 1;
			t_sample sample_storage_147 = m_storage_3.read(index_wrap_18, 0);
			t_sample index_storage_148 = (plusequals_130 + latch_139);
			int eq_131 = (change_133 == ((int)1));
			t_sample plusequals_129 = __m_pluseq_19.post(m_shift_2, eq_131, 0);
			__m_latch_20 = ((eq_131 != 0) ? sub_126 : __m_latch_20);
			t_sample latch_128 = __m_latch_20;
			int index_trunc_21 = fixnan(floor((latch_128 + plusequals_129)));
			int index_wrap_22 = ((index_trunc_21 < 0) ? ((storage_dim - 1) + ((index_trunc_21 + 1) % storage_dim)) : (index_trunc_21 % storage_dim));
			// samples storage channel 1;
			t_sample sample_storage_137 = m_storage_3.read(index_wrap_22, 0);
			t_sample index_storage_138 = (latch_128 + plusequals_129);
			t_sample mix_153 = (sample_storage_147 + (mul_142 * (sample_storage_137 - sample_storage_147)));
			t_sample out1 = mix_153;
			// assign results to output buffer;
			(*(__out1++)) = out1;
			
		};
		return __exception;
		
	};
	inline void set_window(t_param _value) {
		m_window_1 = (_value < 0 ? 0 : (_value > 1 ? 1 : _value));
	};
	inline void set_shift(t_param _value) {
		m_shift_2 = (_value < 0 ? 0 : (_value > 1 ? 1 : _value));
	};
	inline void set_storage(void * _value) {
		m_storage_3.setbuffer(_value);
	};
	
} State;


///
///	Configuration for the genlib API
///

/// Number of signal inputs and outputs

int gen_kernel_numins = 1;
int gen_kernel_numouts = 1;

int num_inputs() { return gen_kernel_numins; }
int num_outputs() { return gen_kernel_numouts; }
int num_params() { return 3; }

/// Assistive lables for the signal inputs and outputs

const char *gen_kernel_innames[] = { "in1" };
const char *gen_kernel_outnames[] = { "out1" };

/// Invoke the signal process of a State object

int perform(CommonState *cself, t_sample **ins, long numins, t_sample **outs, long numouts, long n) {
	State* self = (State *)cself;
	return self->perform(ins, outs, n);
}

/// Reset all parameters and stateful operators of a State object

void reset(CommonState *cself) {
	State* self = (State *)cself;
	self->reset(cself->sr, cself->vs);
}

/// Set a parameter of a State object

void setparameter(CommonState *cself, long index, t_param value, void *ref) {
	State *self = (State *)cself;
	switch (index) {
		case 0: self->set_shift(value); break;
		case 1: self->set_storage(ref); break;
		case 2: self->set_window(value); break;
		
		default: break;
	}
}

/// Get the value of a parameter of a State object

void getparameter(CommonState *cself, long index, t_param *value) {
	State *self = (State *)cself;
	switch (index) {
		case 0: *value = self->m_shift_2; break;
		
		case 2: *value = self->m_window_1; break;
		
		default: break;
	}
}

/// Get the name of a parameter of a State object

const char *getparametername(CommonState *cself, long index) {
	if (index >= 0 && index < cself->numparams) {
		return cself->params[index].name;
	}
	return 0;
}

/// Get the minimum value of a parameter of a State object

t_param getparametermin(CommonState *cself, long index) {
	if (index >= 0 && index < cself->numparams) {
		return cself->params[index].outputmin;
	}
	return 0;
}

/// Get the maximum value of a parameter of a State object

t_param getparametermax(CommonState *cself, long index) {
	if (index >= 0 && index < cself->numparams) {
		return cself->params[index].outputmax;
	}
	return 0;
}

/// Get parameter of a State object has a minimum and maximum value

char getparameterhasminmax(CommonState *cself, long index) {
	if (index >= 0 && index < cself->numparams) {
		return cself->params[index].hasminmax;
	}
	return 0;
}

/// Get the units of a parameter of a State object

const char *getparameterunits(CommonState *cself, long index) {
	if (index >= 0 && index < cself->numparams) {
		return cself->params[index].units;
	}
	return 0;
}

/// Get the size of the state of all parameters of a State object

size_t getstatesize(CommonState *cself) {
	return genlib_getstatesize(cself, &getparameter);
}

/// Get the state of all parameters of a State object

short getstate(CommonState *cself, char *state) {
	return genlib_getstate(cself, state, &getparameter);
}

/// set the state of all parameters of a State object

short setstate(CommonState *cself, const char *state) {
	return genlib_setstate(cself, state, &setparameter);
}

/// Allocate and configure a new State object and it's internal CommonState:

void *create(t_param sr, long vs) {
	State *self = new State;
	self->reset(sr, vs);
	ParamInfo *pi;
	self->__commonstate.inputnames = gen_kernel_innames;
	self->__commonstate.outputnames = gen_kernel_outnames;
	self->__commonstate.numins = gen_kernel_numins;
	self->__commonstate.numouts = gen_kernel_numouts;
	self->__commonstate.sr = sr;
	self->__commonstate.vs = vs;
	self->__commonstate.params = (ParamInfo *)genlib_sysmem_newptr(3 * sizeof(ParamInfo));
	self->__commonstate.numparams = 3;
	// initialize parameter 0 ("m_shift_2")
	pi = self->__commonstate.params + 0;
	pi->name = "shift";
	pi->paramtype = GENLIB_PARAMTYPE_FLOAT;
	pi->defaultvalue = self->m_shift_2;
	pi->defaultref = 0;
	pi->hasinputminmax = false;
	pi->inputmin = 0;
	pi->inputmax = 1;
	pi->hasminmax = true;
	pi->outputmin = 0;
	pi->outputmax = 1;
	pi->exp = 0;
	pi->units = "";		// no units defined
	// initialize parameter 1 ("m_storage_3")
	pi = self->__commonstate.params + 1;
	pi->name = "storage";
	pi->paramtype = GENLIB_PARAMTYPE_SYM;
	pi->defaultvalue = 0.;
	pi->defaultref = 0;
	pi->hasinputminmax = false;
	pi->inputmin = 0;
	pi->inputmax = 1;
	pi->hasminmax = false;
	pi->outputmin = 0;
	pi->outputmax = 1;
	pi->exp = 0;
	pi->units = "";		// no units defined
	// initialize parameter 2 ("m_window_1")
	pi = self->__commonstate.params + 2;
	pi->name = "window";
	pi->paramtype = GENLIB_PARAMTYPE_FLOAT;
	pi->defaultvalue = self->m_window_1;
	pi->defaultref = 0;
	pi->hasinputminmax = false;
	pi->inputmin = 0;
	pi->inputmax = 1;
	pi->hasminmax = true;
	pi->outputmin = 0;
	pi->outputmax = 1;
	pi->exp = 0;
	pi->units = "";		// no units defined
	
	return self;
}

/// Release all resources and memory used by a State object:

void destroy(CommonState *cself) {
	State *self = (State *)cself;
	genlib_sysmem_freeptr(cself->params);
		
	delete self;
}


} // gen_exported::
