from keras.callbacks import ModelCheckpoint

# trains the model
# defined 1 callback: checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, batch_size, train_size, dev_size, epochs=3):
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_generator, steps_per_epoch=train_size//(batch_size//3), validation_data=validation_generator, validation_steps=dev_size//batch_size, epochs=epochs, callbacks=[checkpoint_callback], )

    model.save('model_final')
