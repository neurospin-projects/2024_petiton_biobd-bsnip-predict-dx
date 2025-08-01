
cd /neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/2022_cambroise_surfaugment/experiments
export BHBDATADIR=/neurospin/brainomics/20_deepint/data/fetchers/openbhb/ico5/
export OUTDIR=/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/2022_cambroise_surfaugment/test3107/


# /!\ Long run /!\ Launch the training of a SimCLR-SCNN with Base + GroupMixUp
# augmentations 
python train_ssl.py --datadir $BHBDATADIR --outdir $OUTDIR --latent-dim 128 --batch-size 1024 --normalize --standardize --cutout --blur --noise --learning-rate 2e-3 --epochs 400 --loss-param 2 --groupmixup 0.4



source /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new/bin/activate
python --version  # should show 3.10.13
which python
echo "3.10.13" > /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new/.python-version


vendredi :
rm -rf /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new
python -m venv /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new
source /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new/bin/activate
pip install --upgrade pip setuptools wheel

python -c "import _ctypes; print('ctypes OK')"





source /neurospin/psy_sbox/temp_sara/venv-sbm_dl_new/bin/activate

python -c "import scipy; print(scipy.__version__)"

pip install git+https://github.com/neurospin-deepinsight/surfify@master
python -c "import scipy; import sklearn; import torch; import surfify; print('All imports OK')"



