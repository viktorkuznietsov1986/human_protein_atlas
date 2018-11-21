from keras.callbacks import ModelCheckpoint

# trains the model
# defined 1 callback: checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, epochs=20, use_multiprocessing=True, workers=4, ):
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_generator, validation_data=validation_generator, epochs=epochs,
                        use_multiprocessing=use_multiprocessing, workers=workers,
                        callbacks=[checkpoint_callback], )

    model.save('model_final')
