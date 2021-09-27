from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_50((96, 96, 96, 1), 20)
model.compile(loss="mean_squared_error", optimizer="sgd")
model.fit(X_train, y_train, batch_size=10)