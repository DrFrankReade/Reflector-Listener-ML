# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
import numpy as np
from .artifacts import canonical_hash

def model_features(model, pcm):
    """75 causal features, all derived solely from decoded data and PCM."""
    w0 = float(model.w0)
    if not np.isfinite(w0) or w0 <= 0 or not 1 <= model.L <= 56:
        return np.zeros(75, np.float32)
    f0 = w0 * 8000 / (2 * np.pi)
    freq = np.arange(1, model.L + 1) * f0
    amp = np.array(list(model.Ml)[1:model.L + 1], dtype=np.float64)
    voiced = np.array(list(model.Vl)[1:model.L + 1]) == 1
    amp[~voiced] /= 0.2046 / np.sqrt(w0)
    power = amp * amp
    use = (freq >= 100) & (freq < 3900)
    total = float(power[use].sum())
    if total <= 1e-20 or not np.isfinite(total):
        return np.zeros(75, np.float32)
    edges = np.linspace(100, 3900, 33)
    energy = np.histogram(freq[use], edges, weights=power[use])[0]
    vp = np.histogram(freq[use], edges, weights=power[use] * voiced[use])[0]
    logbands = np.log(np.maximum(energy / total, 1e-12))
    vbands = vp / np.maximum(energy, total * 1e-12)
    use = (freq >= 300) & (freq < 3900)
    f, p, v = freq[use], power[use], voiced[use]
    t = max(float(p.sum()), 1e-20)
    coarse = [p[(f >= lo) & (f < hi)].sum() for lo, hi in ((300,800),(800,1500),(1500,2500),(2500,3900))]
    flatness = np.mean(np.log(np.maximum(p, t * 1e-12))) - np.log(t / max(len(p), 1)) if len(p) else -20
    vf = float(p[v].sum() / t)
    upper = f >= 2500
    uv = float(p[upper & v].sum() / max(float(p[upper].sum()), t * 1e-12))
    old = np.r_[1, np.log(np.maximum(np.array(coarse) / t, 1e-12)), np.clip(flatness, -20, 0),
                vf, uv, vf * np.log(max(f0, 1) / 150), np.sum(p * f) / (4000 * t)]
    level = np.log(max(float(np.mean(np.square(np.asarray(pcm, np.float64) / 32768))), 1e-12))
    features = np.r_[logbands, vbands, old, level].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite decoded features")
    return features

STOCK_SHA = '698cb4ef2ef78ec0528e5ebddee3b5ba6ef964ba7f847ec99bfeb0459bcf11ea'


SEED = 12345


HARMONICS = 56


SCALAR_NAMES = (
    'pre_log_f0', 'post_log_f0', 'pre_L_fraction', 'post_L_fraction',
    'pre_K_fraction', 'post_K_fraction', 'pre_gamma', 'post_gamma',
    'pcm_log_power', 'pre_log_harmonic_power', 'post_log_harmonic_power',
    'pre_model_valid', 'post_model_valid', 'pre_voiced_fraction', 'post_voiced_fraction',
    'pre_pitch_valid', 'post_pitch_valid', 'c0_errors', 'protected_errors',
    'c4_errors', 'total_errors', 'error_rate', 'repeat_count', 'error_count_total',
    'error_count4', 'flag_soft_input', 'flag_c0_valid', 'flag_c4_valid',
    'flag_tone', 'flag_erasure', 'flag_repeat', 'flag_mute', 'flag_silence',
    'flag_c0_uncorrectable', 'flag_python_silence',
)


STATE_DIM = 336 + len(SCALAR_NAMES)


CONTINUOUS_COLUMNS = tuple(range(112)) + tuple(336+i for i in (*range(11),13,14,*range(17,25)))


FLAG_BITS = (1,2,4,0x10,0x20,0x40,0x80,0x100,0x400,0x80000000)


CONTROL_NAMES = ('w0','L','K','gamma','errorRate','repeatCount','errorCountTotal','errorCount4')


SCHEMA = {
    'kind':'stock265_decoder_state_v4', 'version':1, 'conditioning_dimension':STATE_DIM,
    'frame_samples8':160, 'harmonics':56, 'rng_seed_per_utterance':SEED,
    'conditioning_slices':{'pre_log_amplitude':[0,56], 'post_log_amplitude':[56,112],
        'pre_voicing':[112,168], 'post_voicing':[168,224],
        'pre_harmonic_valid':[224,280], 'post_harmonic_valid':[280,336], 'scalars':[336,371]},
    'scalar_names':list(SCALAR_NAMES), 'continuous_columns':list(CONTINUOUS_COLUMNS),
    'raw_control_names':list(CONTROL_NAMES),
    'amplitude_transform':'ln(1 + native Ml); absolute native scale retained',
    'pitch_transform':'ln(decoder F0/150); zero unless a valid model has voiced harmonics',
    'power_transform':'ln(mean/sum power); normalized by32768; PCM uses frozen7/32768 scale',
    'pre_state':'prev_mp after frame: pre-enhancement prediction model for ordinary accepted speech',
    'post_state':'prev_mp_enhanced after frame: enhanced synthesis model for ordinary accepted speech',
    'invalid_model_classes':'tone,erasure,repeat,mute,silence,C0_uncorrectable,Python b0=124 silence',
    'raw_state_note':'Retained raw snapshots may be stale on nonvoice classes; masks are authoritative. Synthesis phases are not original-source phase.',
    'frame_layout':'conditioning float32 [frames,371]; raw per-harmonic arrays use harmonic indices1..56',
    'source_reference_in_conditioning':False,
}


def conditioning_mask(features):
    """Mask invalid acoustic columns; retain explicit status and PCM energy."""
    x=np.asarray(features)
    if x.ndim!=2 or x.shape[1]!=STATE_DIM or not np.isfinite(x).all():
        raise ValueError('Expected finite decoder conditioning [frames,371]')
    mask=np.ones(x.shape,dtype=bool)
    bad=np.any(x[:,364:371]>.5,axis=1)
    pre=(x[:,347]>.5)&~bad; post=(x[:,348]>.5)&~bad
    preh=(x[:,224:280]>.5)&pre[:,None]; posth=(x[:,280:336]>.5)&post[:,None]
    mask[:,0:56]=preh; mask[:,56:112]=posth
    mask[:,112:168]=preh; mask[:,168:224]=posth
    mask[:,224:280]=pre[:,None]; mask[:,280:336]=post[:,None]
    mask[:,347:349]=(~bad)[:,None]
    mask[:,351]=pre; mask[:,352]=post
    mask[:,336]=(x[:,351]>.5)&pre; mask[:,337]=(x[:,352]>.5)&post
    for i in (2,4,6,9,13):mask[:,336+i]=pre
    for i in (3,5,7,10,14):mask[:,336+i]=post
    return mask


def normalize_features(features,normalization):
    if normalization.get('schema_sha256')!=canonical_hash(SCHEMA):raise ValueError('State normalization schema differs')
    if normalization.get('continuous_columns')!=list(CONTINUOUS_COLUMNS):raise ValueError('State normalization columns differ')
    x=np.asarray(features,np.float32); valid=conditioning_mask(x)
    mean=np.asarray(normalization['mean'],np.float32); std=np.asarray(normalization['std'],np.float32)
    if mean.shape!=(STATE_DIM,) or std.shape!=(STATE_DIM,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid state normalization statistics')
    result=x.copy(); columns=np.asarray(CONTINUOUS_COLUMNS)
    result[:,columns]=np.clip((x[:,columns]-mean[columns])/std[columns],-12,12)
    result[~valid]=0
    return result


def _snapshot(model):
    return {**{name:np.asarray(list(getattr(model,name))[1:57],np.float32) for name in ('Ml','log2Ml','PHIl','PSIl')},
            'Vl':np.asarray(list(model.Vl)[1:57],np.int32),
            'control':np.asarray([getattr(model,name) for name in CONTROL_NAMES],np.float32)}


def frame_conditioning(pre,post,current,waveform,result,flags):
    """Pure transformation of snapshots; native synthesis is never invoked."""
    invalid=bool(flags & (0x4f0|0x100|0x80000000))
    def model_values(snapshot):
        control=snapshot['control']; w0,L,K,gamma=control[:4]
        valid=not invalid and np.isfinite(control[:4]).all() and 0<w0<=np.pi and 1<=L<=56 and int(L)==L
        hmask=(np.arange(56)<int(L)) if valid else np.zeros(56,bool)
        ml=snapshot['Ml']; v=snapshot['Vl']
        if valid and (not np.isfinite(ml[hmask]).all() or np.any(ml[hmask]<0) or not np.isin(v[hmask],[0,1]).all()):
            raise ValueError('Invalid valid-harmonic native state')
        amp=np.where(hmask,np.log1p(np.maximum(ml,0)),0).astype(np.float32)
        voiced=(v==1)&hmask; pitch=bool(valid and np.any(voiced))
        f0=float(w0)*8000/(2*np.pi) if valid else 0
        return {'amp':amp,'voice':voiced.astype(np.float32),'mask':hmask.astype(np.float32),
                'valid':float(valid),'pitch':float(pitch),'f0':np.log(f0/150) if pitch else 0,
                'L':float(L)/56 if valid else 0,'K':float(K)/56 if valid else 0,'gamma':float(gamma) if valid else 0,
                'power':np.log(max(float(np.sum((ml[hmask].astype(np.float64)/32768)**2)),1e-20)) if valid else 0,
                'voiced_fraction':float(voiced.sum()/max(hmask.sum(),1))}
    a,b=model_values(pre),model_values(post)
    pcm=np.asarray(waveform,np.float64)*(7/32768)
    controls=current['control']
    values=[a['f0'],b['f0'],a['L'],b['L'],a['K'],b['K'],a['gamma'],b['gamma'],
        np.log(max(float(np.mean(pcm*pcm)),1e-20)),a['power'],b['power'],a['valid'],b['valid'],
        a['voiced_fraction'],b['voiced_fraction'],a['pitch'],b['pitch'],
        result.c0_errors,result.protected_errors,result.c4_errors,result.total_errors,
        controls[4],controls[5],controls[6],controls[7],*[float(bool(flags&bit)) for bit in FLAG_BITS]]
    vector=np.r_[a['amp'],b['amp'],a['voice'],b['voice'],a['mask'],b['mask'],values].astype(np.float32)
    if vector.shape!=(STATE_DIM,) or not np.isfinite(vector).all():raise ValueError('Nonfinite state conditioning')
    return vector
